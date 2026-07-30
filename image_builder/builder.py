from __future__ import annotations

import contextlib
import os
import secrets
import shutil
import signal
import tempfile
import threading
import warnings
from pathlib import Path
from typing import Callable, Sequence

from .contracts import (
    INPUT_ROLES,
    RESTORED_FILESYSTEM_SPECS,
)
from .errors import BuilderError
from .extfs import (
    P7_SKELETON_SHA256,
    create_legacy_ext,
    create_swap,
    extract_syslinux_mbr,
    parse_superblock,
    populate_p7,
    run_e2fsck,
    run_e2fsck_path,
    validate_superblock,
)
from .geometry import (
    DISK_BYTES,
    PARTITION_BY_NUMBER,
    SECTOR_SIZE,
    write_table,
)
from .features import (
    FeatureCatalog,
    FeatureInstallPlan,
    apply_feature_install,
    feature_patch_target_paths,
    feature_target_paths,
    plan_feature_install,
)
from .io_utils import (
    OpenedRegular,
    copy_sparse_to_offsets,
    open_regular_read,
    publish_no_replace,
    sha256_fd,
    validate_new_output_path,
)
from .jukebox import (
    JUKEBOX_TARGET_PATH,
    stage_jukebox_fix,
)
from .verifier import _verify_assembled_image

Progress = Callable[[str], None]
MIN_BUILD_FREE_BYTES = 24 * 1024 * 1024 * 1024
PREFLIGHT_SPARSE_BYTES = 64 * 1024 * 1024


def _validate_prepared_filesystems(
    boot_value: str | os.PathLike[str],
    root_value: str | os.PathLike[str],
    vr_value: str | os.PathLike[str],
    *,
    progress: Progress | None = None,
) -> tuple[dict[str, object], contextlib.ExitStack, dict[str, OpenedRegular]]:
    stack = contextlib.ExitStack()
    opened: dict[str, OpenedRegular] = {}
    try:
        values = {"boot": boot_value, "root": root_value, "vr": vr_value}
        for role in INPUT_ROLES:
            opened[role] = stack.enter_context(open_regular_read(values[role]))
        report: dict[str, object] = {
            "schema": "genesis-partition-validation/v1",
            "stage": "source",
            "valid": True,
            "partitions": {},
        }
        for role in INPUT_ROLES:
            item = opened[role]
            spec = RESTORED_FILESYSTEM_SPECS[role]
            expected_size = int(spec["bytes"])
            if item.size != expected_size:
                raise BuilderError(
                    f"{role} filesystem is {item.size} bytes, expected {expected_size}"
                )
            if progress:
                progress(f"hashing restored {role} filesystem ({item.size} bytes)")
            digest = sha256_fd(item.fd, item.size)
            superblock = parse_superblock(item.fd)
            validate_superblock(
                superblock,
                spec,
                description=role,
                container_bytes=item.size,
            )
            if progress:
                progress(f"running non-writing e2fsck on restored {role}")
            run_e2fsck(item, f"restored {role}")
            item.assert_unchanged()
            partitions = report["partitions"]
            if not isinstance(partitions, dict):
                raise BuilderError("internal partition-validation report is invalid")
            partitions[role] = {
                "bytes": item.size,
                "sha256": digest,
                "filesystem": superblock.as_dict(),
            }
        return report, stack, opened
    except Exception:
        stack.close()
        raise


def _assemble_prepared_image(
    *,
    boot_path: str | os.PathLike[str],
    root_path: str | os.PathLike[str],
    vr_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    feature_catalog: FeatureCatalog | None = None,
    selected_feature_ids: Sequence[str] = (),
    selected_public_patch_ids: Sequence[str] = (),
    excluded_feature_ids: Sequence[str] = (),
    excluded_public_patch_ids: Sequence[str] = (),
    progress: Progress | None = None,
) -> dict[str, object]:
    if os.geteuid() == 0:
        raise BuilderError(
            "refusing to build as root; this version only needs ordinary user access"
        )
    if feature_catalog is None and (
        selected_feature_ids
        or selected_public_patch_ids
        or excluded_feature_ids
        or excluded_public_patch_ids
    ):
        raise BuilderError("feature selections require a loaded feature catalog")
    output = validate_new_output_path(output_path)
    if progress:
        progress("preflighting sparse-file, hardlink, and free-space support")
    _preflight_output_directory(output.parent)
    source_validation, stack, opened = _validate_prepared_filesystems(
        boot_path,
        root_path,
        vr_path,
        progress=progress,
    )
    source_partitions = source_validation.get("partitions")
    if not isinstance(source_partitions, dict):
        stack.close()
        raise BuilderError("partition validation returned no source records")
    source_hashes: dict[str, str] = {}
    for role in INPUT_ROLES:
        record = source_partitions.get(role)
        digest = record.get("sha256") if isinstance(record, dict) else None
        if not isinstance(digest, str):
            stack.close()
            raise BuilderError(f"partition validation returned no {role} hash")
        source_hashes[role] = digest
    workdir: Path | None = None
    image_fd = -1
    previous_sigterm: object | None = None
    try:
        feature_plan: FeatureInstallPlan | None = None
        if feature_catalog is not None and (
            selected_feature_ids or selected_public_patch_ids
        ):
            feature_plan = plan_feature_install(
                opened["root"].path,
                feature_catalog,
                selected_feature_ids=selected_feature_ids,
                selected_public_patch_ids=selected_public_patch_ids,
                excluded_feature_ids=excluded_feature_ids,
                excluded_public_patch_ids=excluded_public_patch_ids,
            )
            feature_files = feature_target_paths(feature_plan)
            feature_patches = feature_patch_target_paths(feature_plan)
            if JUKEBOX_TARGET_PATH in feature_files | feature_patches:
                raise BuilderError(
                    "local catalog targets the mandatory dynamic Jukebox "
                    "correction file: " + JUKEBOX_TARGET_PATH
                )
        workdir = Path(
            tempfile.mkdtemp(prefix=".genesis-build-", dir=os.fspath(output.parent))
        )
        os.chmod(workdir, 0o700)
        previous_sigterm = _install_sigterm_handler()
        temporary_image = workdir / "image.partial"
        image_fd = os.open(
            temporary_image,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o600,
        )
        os.ftruncate(image_fd, DISK_BYTES)
        disk_id = _new_disk_id()
        bootstrap = extract_syslinux_mbr(opened["root"])
        write_table(image_fd, bootstrap, disk_id)
        if progress:
            progress("created exact canonical MBR and EBR chain")

        swap_path = workdir / "p3.swap"
        create_swap(swap_path, PARTITION_BY_NUMBER[3].bytes)
        fresh_paths: dict[int, Path] = {}
        p7_manifest_hash: str | None = None
        p10_superblock = None
        for number in (7, 8, 9, 10, 12, 13):
            if progress:
                progress(f"creating legacy-compatible fresh p{number}")
            partition_path = workdir / f"p{number}.ext4"
            superblock = create_legacy_ext(
                partition_path, PARTITION_BY_NUMBER[number].bytes, number
            )
            if number == 7:
                p7_manifest_hash = populate_p7(partition_path)
                if p7_manifest_hash != P7_SKELETON_SHA256:
                    raise BuilderError("p7 skeleton changed while the build was running")
            if number == 10:
                p10_superblock = superblock
            run_e2fsck_path(partition_path, f"fresh p{number}")
            fresh_paths[number] = partition_path
        if p7_manifest_hash is None:
            raise BuilderError("p7 skeleton was not created")
        if p10_superblock is None:
            raise BuilderError("p10 filesystem was not created")

        customized_root_path = workdir / "root.customized.ext4"
        staged_root, jukebox_storage = stage_jukebox_fix(
            opened["root"],
            customized_root_path,
            p10_superblock,
            progress=progress,
        )
        staged_root.close()
        feature_install_report: dict[str, object] | None = None
        if feature_plan is not None:
            feature_install_report = apply_feature_install(
                customized_root_path,
                feature_plan,
                progress=progress,
            )
        staged_root = open_regular_read(customized_root_path)
        root_payload = stack.enter_context(staged_root)
        if progress:
            progress("hashing customized root filesystem")
        root_payload_sha256 = sha256_fd(root_payload.fd, root_payload.size)

        payload_plan = {
            "boot": (opened["boot"], (1, 5)),
            "root": (root_payload, (2, 6)),
            "vr": (opened["vr"], (11,)),
        }
        for role, (source, partition_numbers) in payload_plan.items():
            if progress:
                destinations = ", ".join(f"p{number}" for number in partition_numbers)
                progress(f"copying restored {role} into {destinations}")
            offsets = [
                PARTITION_BY_NUMBER[number].start * SECTOR_SIZE
                for number in partition_numbers
            ]
            copy_sparse_to_offsets(
                source.fd, image_fd, source.size, offsets
            )
            source.assert_unchanged()

        with open_regular_read(swap_path) as swap_input:
            copy_sparse_to_offsets(
                swap_input.fd,
                image_fd,
                swap_input.size,
                (PARTITION_BY_NUMBER[3].start * SECTOR_SIZE,),
            )
        for number, path in fresh_paths.items():
            with open_regular_read(path) as fresh_input:
                copy_sparse_to_offsets(
                    fresh_input.fd,
                    image_fd,
                    fresh_input.size,
                    (PARTITION_BY_NUMBER[number].start * SECTOR_SIZE,),
                )
                fresh_input.assert_unchanged()
        os.fsync(image_fd)
        if progress:
            progress("verifying partitions after applying modifications")
        image_verification = _verify_assembled_image(
            image_fd,
            DISK_BYTES,
            expected_payload_sha256={
                **source_hashes,
                "root": root_payload_sha256,
            },
            progress=progress,
        )
        verified_table = image_verification.get("table")
        if (
            not isinstance(verified_table, dict)
            or not isinstance(verified_table.get("disk_id"), str)
        ):
            raise BuilderError("partition verification returned no disk ID")
        verified_disk_id = verified_table["disk_id"]
        os.close(image_fd)
        image_fd = -1
        publish_no_replace(temporary_image, output)
        return {
            "schema": "genesis-image-build-result/v11",
            "built": True,
            "output": os.fspath(output),
            "source_validation": source_validation,
            "image_verification": image_verification,
            "jukebox_storage": jukebox_storage,
            "feature_install": feature_install_report,
            "image_bytes": DISK_BYTES,
            "disk_id": verified_disk_id,
        }
    finally:
        _restore_sigterm_handler(previous_sigterm)
        if image_fd >= 0:
            os.close(image_fd)
        stack.close()
        if workdir is not None:
            try:
                shutil.rmtree(workdir)
            except FileNotFoundError:
                pass
            except OSError as exc:
                message = (
                    f"temporary build directory remains at {workdir}: "
                    f"{exc.strerror}; inspect it before removing it manually"
                )
                if progress:
                    progress(f"WARNING: {message}")
                else:
                    warnings.warn(message, RuntimeWarning, stacklevel=2)


def _new_disk_id() -> int:
    while True:
        value = secrets.randbits(32)
        if value not in (0, 0xFFFFFFFF):
            return value


def _preflight_output_directory(parent: Path) -> None:
    stale = []
    try:
        entries = list(parent.iterdir())
    except OSError as exc:
        raise BuilderError(
            f"cannot scan output directory {parent}: {exc.strerror}"
        ) from exc
    for entry in entries:
        if not entry.name.startswith(".genesis-build-"):
            continue
        try:
            if entry.is_dir() and not entry.is_symlink():
                stale.append(entry)
        except OSError:
            stale.append(entry)
    if stale:
        rendered = ", ".join(os.fspath(path) for path in sorted(stale))
        raise BuilderError(
            "stale or active Genesis build directories need inspection before "
            f"continuing: {rendered}"
        )
    available = shutil.disk_usage(parent).free
    if available < MIN_BUILD_FREE_BYTES:
        raise BuilderError(
            f"output filesystem has {available} free bytes; at least "
            f"{MIN_BUILD_FREE_BYTES} are required for a safe build"
        )
    token = secrets.token_hex(12)
    sparse_path = parent / f".genesis-preflight-{token}.sparse"
    link_path = parent / f".genesis-preflight-{token}.link"
    fd = -1
    try:
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        flags |= getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(sparse_path, flags, 0o600)
        os.ftruncate(fd, PREFLIGHT_SPARSE_BYTES)
        os.pwrite(fd, b"G", 0)
        os.pwrite(fd, b"R", PREFLIGHT_SPARSE_BYTES - 1)
        os.fsync(fd)
        allocation = os.fstat(fd).st_blocks * 512
        if allocation >= PREFLIGHT_SPARSE_BYTES // 2:
            raise BuilderError(
                "output filesystem does not preserve sparse files efficiently"
            )
        os.link(sparse_path, link_path, follow_symlinks=False)
        source_stat = os.stat(sparse_path, follow_symlinks=False)
        link_stat = os.stat(link_path, follow_symlinks=False)
        if (source_stat.st_dev, source_stat.st_ino) != (
            link_stat.st_dev,
            link_stat.st_ino,
        ):
            raise BuilderError("output filesystem hardlink preflight was inconsistent")
    except BuilderError:
        raise
    except OSError as exc:
        raise BuilderError(
            f"output filesystem preflight failed in {parent}: {exc.strerror}"
        ) from exc
    finally:
        if fd >= 0:
            os.close(fd)
        for path in (link_path, sparse_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                warnings.warn(
                    f"preflight artifact remains at {path}: {exc.strerror}",
                    RuntimeWarning,
                    stacklevel=2,
                )


def _install_sigterm_handler() -> object | None:
    if threading.current_thread() is not threading.main_thread():
        return None
    previous = signal.getsignal(signal.SIGTERM)

    def interrupted(_signum: int, _frame: object) -> None:
        raise BuilderError("build interrupted by SIGTERM")

    signal.signal(signal.SIGTERM, interrupted)
    return previous


def _restore_sigterm_handler(previous: object | None) -> None:
    if previous is None or threading.current_thread() is not threading.main_thread():
        return
    signal.signal(signal.SIGTERM, previous)
