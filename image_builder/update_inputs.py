from __future__ import annotations

import contextlib
import os
import re
import shlex
import shutil
import stat
import subprocess
import tempfile
import zlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Final

from .contracts import INPUT_ROLES, RESTORED_FILESYSTEM_SPECS
from .errors import BuilderError
from .features import load_feature_catalog
from .io_utils import (
    OpenedRegular,
    open_regular_read,
    validate_new_output_path,
)

Progress = Callable[[str], None]

UPDATE_KEY_BYTES: Final = 16
UPDATE_KEY_PATH: Final = (
    Path(__file__).resolve().parent / "resources" / "genesis.update-key"
)
CATALOG_MAX_BYTES: Final = 1024 * 1024
CRC_CHUNK: Final = 4 * 1024 * 1024
COMMAND_LOG_TAIL: Final = 32 * 1024
PARTCLONE_MAGIC: Final = b"partclone-image"
MIN_UPDATE_BUILD_FREE_BYTES: Final = 32 * 1024 * 1024 * 1024
DECODER_SOURCE: Final = (
    Path(__file__).resolve().parent / "resources" / "decrypt_lg_container.c"
)
LOCAL_CATALOG_ROOT: Final = Path(__file__).resolve().parent.parent

UPDATE_PATHS: Final = {
    "boot": Path("HU/images/boot.img"),
    "root": Path("HU/images/NP/rootfs.img"),
    "vr": Path("HU/images/vr.img"),
}

_DECIMAL_RE = re.compile(r"-?(?:0|[1-9][0-9]*)\Z")
_UNSIGNED_DECIMAL_RE = re.compile(r"(?:0|[1-9][0-9]*)\Z")


@dataclass(frozen=True)
class _PreparedUpdateInputs:
    """Restored filesystems and the shipped integrity checks they passed."""

    boot_path: Path
    root_path: Path
    vr_path: Path
    update_verification: dict[str, object]


@dataclass(frozen=True)
class CatalogRecord:
    relative_path: Path
    record_id: int
    crc32_signed: int
    size: int


@dataclass(frozen=True)
class UpdateCatalog:
    package_name: str
    records: dict[Path, CatalogRecord]


@contextlib.contextmanager
def _prepare_update_inputs(
    update_directory: str | os.PathLike[str],
    *,
    work_directory: str | os.PathLike[str],
    progress: Progress | None = None,
) -> Iterator[_PreparedUpdateInputs]:
    """Validate, decrypt, and restore the three official update images.

    All generated files live in a private temporary directory below
    ``work_directory`` and are removed when the context exits. Source files
    are opened read-only and checked for replacement while they are in use.
    """

    update_root = _require_directory(update_directory, "update directory")
    work_root = _require_directory(work_directory, "work directory")
    update_real = _require_work_outside_update_tree(update_root, work_root)
    catalog_path = _single_catalog_path(update_root)

    with contextlib.ExitStack() as stack:
        catalog_file = stack.enter_context(open_regular_read(catalog_path))
        if not 0 < catalog_file.size <= CATALOG_MAX_BYTES:
            raise BuilderError(
                f"update catalog is {catalog_file.size} bytes; "
                f"expected 1..{CATALOG_MAX_BYTES}"
            )
        catalog_bytes = _read_exact(catalog_file)
        catalog = parse_update_catalog(
            catalog_bytes,
            catalog_filename=catalog_path.name,
        )

        update_key = stack.enter_context(open_regular_read(UPDATE_KEY_PATH))
        if update_key.size != UPDATE_KEY_BYTES:
            raise BuilderError(
                f"update key is {update_key.size} bytes, "
                f"expected {UPDATE_KEY_BYTES}"
            )
        update_key.assert_unchanged()

        sources: dict[str, OpenedRegular] = {}
        verified_images: dict[str, dict[str, object]] = {}
        for role in INPUT_ROLES:
            relative = UPDATE_PATHS[role]
            source = stack.enter_context(
                _open_update_regular(update_root, relative)
            )
            sources[role] = source
            record = catalog.records.get(relative)
            if record is None:
                raise BuilderError(
                    f"update catalog does not contain {relative.as_posix()}"
                )
            if source.size != record.size:
                raise BuilderError(
                    f"catalog size mismatch for {relative.as_posix()}: "
                    f"catalog {record.size}, file {source.size}"
                )
            if progress:
                progress(
                    f"validating {relative.as_posix()} against "
                    f"{catalog_path.name}"
                )
            crc_unsigned = crc32_fd(source.fd, source.size)
            crc_signed = signed_crc32(crc_unsigned)
            if crc_signed != record.crc32_signed:
                raise BuilderError(
                    f"catalog CRC-32 mismatch for {relative.as_posix()}: "
                    f"catalog {record.crc32_signed}, file {crc_signed}"
                )
            source.assert_unchanged()
            verified_images[role] = {
                "path": relative.as_posix(),
                "bytes": source.size,
                "crc32_signed": crc_signed,
            }

        if os.pread(sources["vr"].fd, len(PARTCLONE_MAGIC), 0) != PARTCLONE_MAGIC:
            raise BuilderError("HU/images/vr.img is not a plain Partclone image")

        preparation = Path(
            tempfile.mkdtemp(prefix="prepared-update-", dir=work_root)
        )
        stack.callback(_remove_private_tree, preparation)
        os.chmod(preparation, 0o700)

        tools = {
            "cc": require_host_tool("cc", forbidden_root=update_real),
            "pkg-config": require_host_tool(
                "pkg-config",
                forbidden_root=update_real,
            ),
            "partclone.restore": require_host_tool(
                "partclone.restore",
                forbidden_root=update_real,
            ),
        }
        decoder = compile_decoder(
            preparation,
            cc=tools["cc"],
            pkg_config=tools["pkg-config"],
        )

        restored: dict[str, Path] = {}
        for role in INPUT_ROLES:
            decrypted: Path | None = None
            if role in ("boot", "root"):
                decrypted = preparation / f"{role}.partclone"
                if progress:
                    progress(
                        f"decrypting and validating {role} update container"
                    )
                _run_checked(
                    [
                        str(decoder),
                        "--key",
                        fd_path(update_key.fd),
                        "--input",
                        fd_path(sources[role].fd),
                        "--output",
                        str(decrypted),
                        "--expect-partclone",
                    ],
                    description=f"decrypt {role} update container",
                    log_path=preparation / f"{role}.decrypt.log",
                    pass_fds=(update_key.fd, sources[role].fd),
                )
                _require_private_regular(
                    decrypted,
                    description=f"decrypted {role} Partclone stream",
                )
                stream_path = str(decrypted)
                pass_fds: tuple[int, ...] = ()
            else:
                stream_path = fd_path(sources["vr"].fd)
                pass_fds = (sources["vr"].fd,)

            output = preparation / f"{role}.filesystem"
            if progress:
                progress(
                    f"CRC-checking and restoring {role} filesystem "
                    "to a regular file"
                )
            try:
                _run_checked(
                    [
                        tools["partclone.restore"],
                        "--restore_raw_file",
                        "--source",
                        stream_path,
                        "--output",
                        str(output),
                        "--logfile",
                        str(preparation / f"{role}.partclone-restore.log"),
                        "--quiet",
                    ],
                    description=f"restore {role} Partclone stream",
                    log_path=(
                        preparation / f"{role}.partclone-restore.console.log"
                    ),
                    pass_fds=pass_fds,
                )
            finally:
                if decrypted is not None:
                    try:
                        decrypted.unlink()
                    except FileNotFoundError:
                        pass
            expected_size = int(RESTORED_FILESYSTEM_SPECS[role]["bytes"])
            _require_private_regular(
                output,
                description=f"restored {role} filesystem",
                expected_size=expected_size,
            )
            restored[role] = output

        for item in (catalog_file, update_key, *sources.values()):
            item.assert_unchanged()

        update_verification: dict[str, object] = {
            "schema": "genesis-update-verification/v1",
            "catalog": {
                "filename": catalog_path.name,
                "package_name": catalog.package_name,
            },
            "images": verified_images,
            "checks_passed": {
                "catalog_byte_size_and_crc32": list(INPUT_ROLES),
                "lg_trailer_and_encrypted_payload_sha256": ["boot", "root"],
                "partclone_record_crc": list(INPUT_ROLES),
            },
        }
        yield _PreparedUpdateInputs(
            boot_path=restored["boot"],
            root_path=restored["root"],
            vr_path=restored["vr"],
            update_verification=update_verification,
        )


def build_image_from_update(
    *,
    update_directory: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    apply_default_patches: bool = True,
    excluded_patch_ids: Sequence[str] = (),
    apply_default_features: bool = True,
    excluded_feature_ids: Sequence[str] = (),
    progress: Progress | None = None,
) -> dict[str, object]:
    """Prepare official update files and pass them to the normal image build."""

    from .builder import _assemble_prepared_image

    if os.geteuid() == 0:
        raise BuilderError(
            "refusing to build as root; this version only needs ordinary user access"
        )
    if not apply_default_patches and excluded_patch_ids:
        raise BuilderError(
            "cannot exclude byte patches when local patches are disabled"
        )
    if not apply_default_features and excluded_feature_ids:
        raise BuilderError(
            "cannot exclude feature packages when local features are disabled"
        )
    catalog = load_feature_catalog(LOCAL_CATALOG_ROOT)
    all_patch_ids = tuple(recipe.patch_id for recipe in catalog.public_patches)
    all_feature_ids = tuple(
        definition.feature_id for definition in catalog.features
    )
    if apply_default_patches:
        normalized_patch_exclusions = _normalize_catalog_exclusions(
            excluded_patch_ids,
            all_patch_ids,
            "byte patch",
        )
        selected_public_patch_ids = tuple(
            patch_id
            for patch_id in all_patch_ids
            if patch_id not in normalized_patch_exclusions
        )
    else:
        normalized_patch_exclusions = all_patch_ids
        selected_public_patch_ids = ()
    if apply_default_features:
        normalized_feature_exclusions = _normalize_catalog_exclusions(
            excluded_feature_ids,
            all_feature_ids,
            "feature package",
        )
        selected_feature_ids = tuple(
            feature_id
            for feature_id in all_feature_ids
            if feature_id not in normalized_feature_exclusions
        )
    else:
        normalized_feature_exclusions = all_feature_ids
        selected_feature_ids = ()
    output = validate_new_output_path(output_path)
    output_parent = _require_directory(output.parent, "output directory")
    update_root = _require_directory(update_directory, "update directory")
    _require_work_outside_update_tree(update_root, output_parent)
    _preflight_update_output_directory(output_parent)
    with tempfile.TemporaryDirectory(
        prefix=".genesis-update-build-",
        dir=output_parent,
    ) as temporary:
        os.chmod(temporary, 0o700)
        with _prepare_update_inputs(
            update_root,
            work_directory=temporary,
            progress=progress,
        ) as prepared:
            result = _assemble_prepared_image(
                boot_path=prepared.boot_path,
                root_path=prepared.root_path,
                vr_path=prepared.vr_path,
                output_path=output_path,
                feature_catalog=catalog,
                selected_feature_ids=selected_feature_ids,
                selected_public_patch_ids=selected_public_patch_ids,
                excluded_feature_ids=normalized_feature_exclusions,
                excluded_public_patch_ids=normalized_patch_exclusions,
                progress=progress,
            )
            result["update_verification"] = prepared.update_verification
            return result


def _normalize_catalog_exclusions(
    values: Sequence[str],
    available_ids: Sequence[str],
    description: str,
) -> tuple[str, ...]:
    available = set(available_ids)
    excluded: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value:
            raise BuilderError(f"excluded {description} IDs must be nonempty strings")
        if value in seen:
            raise BuilderError(f"duplicate excluded {description}: {value}")
        if value not in available:
            choices = ", ".join(available_ids) if available_ids else "none"
            raise BuilderError(
                f"unknown excluded {description} {value!r}; "
                f"available {description}s: {choices}"
            )
        seen.add(value)
        excluded.append(value)
    return tuple(excluded)


def _preflight_update_output_directory(output_parent: Path) -> None:
    try:
        stale = sorted(
            entry
            for entry in output_parent.iterdir()
            if entry.name.startswith(".genesis-update-build-")
        )
    except OSError as exc:
        raise BuilderError(
            f"cannot scan output directory {output_parent}: {exc}"
        ) from exc
    if stale:
        raise BuilderError(
            "stale or active update preparation directories need inspection: "
            + ", ".join(str(path) for path in stale)
        )
    available = shutil.disk_usage(output_parent).free
    if available < MIN_UPDATE_BUILD_FREE_BYTES:
        raise BuilderError(
            f"output filesystem has {available} free bytes; encrypted update "
            f"input mode requires at least {MIN_UPDATE_BUILD_FREE_BYTES}"
        )


def parse_update_catalog(
    data: bytes,
    *,
    catalog_filename: str,
) -> UpdateCatalog:
    """Parse a card catalog and safely normalize its package-prefixed paths."""

    if not data or len(data) > CATALOG_MAX_BYTES:
        raise BuilderError("update catalog has an unsupported size")
    try:
        text = data.decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        raise BuilderError("update catalog is not ASCII") from exc
    if "\x00" in text:
        raise BuilderError("update catalog contains a NUL byte")
    lines = text.splitlines()
    if len(lines) < 2:
        raise BuilderError("update catalog has no file records")

    header = lines[0].split("|")
    if len(header) != 7 or header[0] != "+":
        raise BuilderError("update catalog has an invalid header")
    package_name = header[4]
    _require_safe_component(package_name, "catalog package name")
    if Path(catalog_filename).name != catalog_filename:
        raise BuilderError("catalog filename is not a basename")
    if Path(catalog_filename).suffix != ".ver":
        raise BuilderError("update catalog does not have a .ver suffix")
    if Path(catalog_filename).stem != package_name:
        raise BuilderError(
            "update catalog filename does not match its package name"
        )

    records: dict[Path, CatalogRecord] = {}
    record_ids: set[int] = set()
    for line_number, line in enumerate(lines[1:], start=2):
        fields = line.split("|")
        if len(fields) != 6:
            raise BuilderError(
                f"update catalog line {line_number} has an invalid field count"
            )
        directory, filename, record_id_text, crc_text, size_text, flag = fields
        relative = normalize_catalog_path(
            package_name,
            directory,
            filename,
        )
        record_id = _parse_unsigned_decimal(
            record_id_text,
            f"catalog record ID on line {line_number}",
        )
        crc_signed = _parse_signed_crc32(
            crc_text,
            f"catalog CRC-32 on line {line_number}",
        )
        size = _parse_unsigned_decimal(
            size_text,
            f"catalog size on line {line_number}",
        )
        _parse_unsigned_decimal(flag, f"catalog flag on line {line_number}")
        if relative in records:
            raise BuilderError(
                f"update catalog contains duplicate path {relative.as_posix()}"
            )
        if record_id in record_ids:
            raise BuilderError(
                f"update catalog contains duplicate record ID {record_id}"
            )
        record_ids.add(record_id)
        records[relative] = CatalogRecord(
            relative_path=relative,
            record_id=record_id,
            crc32_signed=crc_signed,
            size=size,
        )

    missing = [
        path.as_posix() for path in UPDATE_PATHS.values() if path not in records
    ]
    if missing:
        raise BuilderError(
            "update catalog lacks required image records: " + ", ".join(missing)
        )
    return UpdateCatalog(package_name=package_name, records=records)


def normalize_catalog_path(
    package_name: str,
    directory: str,
    filename: str,
) -> Path:
    """Convert a catalog's Windows directory field into a safe relative path."""

    _require_safe_component(package_name, "catalog package name")
    _require_safe_component(filename, "catalog filename")
    normalized = directory.replace("\\", "/")
    if normalized.startswith("/") or normalized.endswith("/"):
        raise BuilderError(f"unsafe catalog directory: {directory!r}")
    parts = normalized.split("/")
    if (
        not parts
        or any(part in ("", ".", "..") for part in parts)
        or parts[0] != package_name
    ):
        raise BuilderError(f"unsafe catalog directory: {directory!r}")
    relative_parts = parts[1:] + [filename]
    if not relative_parts:
        raise BuilderError(f"unsafe catalog path: {directory!r}|{filename!r}")
    relative = Path(*relative_parts)
    if relative.is_absolute() or ".." in relative.parts:
        raise BuilderError(f"unsafe catalog path: {directory!r}|{filename!r}")
    return relative


def crc32_fd(fd: int, size: int) -> int:
    """Return the standard unsigned CRC-32 in one bounded pass."""

    if size < 0:
        raise BuilderError("cannot CRC-check a negative byte count")
    crc = 0
    position = 0
    while position < size:
        chunk = os.pread(fd, min(CRC_CHUNK, size - position), position)
        if not chunk:
            raise BuilderError(
                f"unexpected EOF while CRC-checking update input at byte {position}"
            )
        crc = zlib.crc32(chunk, crc)
        position += len(chunk)
    return crc & 0xFFFFFFFF


def signed_crc32(value: int) -> int:
    if not 0 <= value <= 0xFFFFFFFF:
        raise BuilderError(f"CRC-32 value is out of range: {value}")
    return value if value < 0x80000000 else value - 0x100000000


def require_host_tool(
    name: str,
    *,
    forbidden_root: Path | None = None,
) -> str:
    """Resolve an executable from the host, never from update-card contents."""

    found = shutil.which(name)
    if found is None:
        raise BuilderError(f"required host tool is unavailable: {name}")
    try:
        resolved = Path(found).resolve(strict=True)
        metadata = resolved.stat()
    except OSError as exc:
        raise BuilderError(f"cannot inspect host tool {name}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode) or not os.access(resolved, os.X_OK):
        raise BuilderError(f"host tool is not an executable regular file: {name}")
    if forbidden_root is not None and resolved.is_relative_to(forbidden_root):
        raise BuilderError(
            f"refusing update-tree executable as host tool: {resolved}"
        )
    return str(resolved)


def compile_decoder(
    work_directory: Path,
    *,
    cc: str,
    pkg_config: str,
) -> Path:
    """Compile the bundled bounded-memory LG container decoder privately."""

    try:
        source_metadata = DECODER_SOURCE.stat()
    except OSError as exc:
        raise BuilderError(f"LG decoder source is unavailable: {exc}") from exc
    if not stat.S_ISREG(source_metadata.st_mode):
        raise BuilderError("LG decoder source is not a regular file")

    flags_output = work_directory / "openssl-pkg-config.out"
    flags_error = work_directory / "openssl-pkg-config.err"
    flags_result = _run_to_private_files(
        [pkg_config, "--cflags", "--libs", "openssl"],
        stdout_path=flags_output,
        stderr_path=flags_error,
    )
    if flags_result != 0:
        detail = _read_log_tail(flags_error)
        raise BuilderError(
            "cannot obtain OpenSSL compiler flags"
            + (f": {detail}" if detail else "")
        )
    try:
        flags_size = flags_output.stat().st_size
    except OSError as exc:
        raise BuilderError(f"cannot inspect OpenSSL compiler flags: {exc}") from exc
    if flags_size > 64 * 1024:
        raise BuilderError("OpenSSL compiler flags output is unexpectedly large")
    try:
        openssl_flags = shlex.split(
            flags_output.read_text(encoding="utf-8", errors="strict")
        )
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise BuilderError("OpenSSL compiler flags are invalid") from exc

    decoder = work_directory / "decrypt_lg_container"
    _run_checked(
        [
            cc,
            "-O2",
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(DECODER_SOURCE),
            "-o",
            str(decoder),
            *openssl_flags,
        ],
        description="compile LG container decoder",
        log_path=work_directory / "decoder-compile.log",
    )
    metadata = decoder.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise BuilderError("compiled LG decoder is not a regular file")
    os.chmod(decoder, 0o700)
    return decoder


def fd_path(fd: int) -> str:
    if fd < 0:
        raise BuilderError("cannot pass a closed input to a host tool")
    return f"/proc/self/fd/{fd}"


def _single_catalog_path(update_root: Path) -> Path:
    try:
        candidates = sorted(
            (
                Path(entry.path)
                for entry in os.scandir(update_root)
                if entry.name.endswith(".ver")
            ),
            key=lambda path: path.name,
        )
    except OSError as exc:
        raise BuilderError(f"cannot inspect update directory: {exc}") from exc
    if len(candidates) != 1:
        raise BuilderError(
            "update directory must contain exactly one root-level .ver catalog"
        )
    try:
        metadata = candidates[0].lstat()
    except OSError as exc:
        raise BuilderError(f"cannot inspect update catalog: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise BuilderError("root-level .ver catalog is not a regular file")
    return candidates[0]


def _open_update_regular(
    update_root: Path,
    relative: Path,
) -> OpenedRegular:
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise BuilderError(f"unsafe update input path: {relative}")
    parent = update_root
    for component in relative.parts[:-1]:
        _require_safe_component(component, "update path component")
        parent = parent / component
        try:
            metadata = parent.lstat()
        except OSError as exc:
            raise BuilderError(f"cannot inspect update path {parent}: {exc}") from exc
        if not stat.S_ISDIR(metadata.st_mode):
            raise BuilderError(
                f"update path component is not a directory: {parent}"
            )
    _require_safe_component(relative.name, "update filename")
    return open_regular_read(parent / relative.name)


def _require_directory(
    value: str | os.PathLike[str],
    description: str,
) -> Path:
    path = Path(value)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise BuilderError(f"cannot inspect {description} {path}: {exc}") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise BuilderError(f"{description} is not a directory: {path}")
    return path


def _require_work_outside_update_tree(
    update_root: Path,
    work_root: Path,
) -> Path:
    update_real = update_root.resolve(strict=True)
    work_real = work_root.resolve(strict=True)
    if work_real == update_real or work_real.is_relative_to(update_real):
        raise BuilderError(
            "update preparation directory must be outside the source update tree"
        )
    return update_real


def _read_exact(opened: OpenedRegular) -> bytes:
    data = os.pread(opened.fd, opened.size, 0)
    if len(data) != opened.size:
        raise BuilderError(f"short read from {opened.path}")
    opened.assert_unchanged()
    return data


def _require_safe_component(value: str, description: str) -> None:
    if (
        value in ("", ".", "..")
        or "/" in value
        or "\\" in value
        or "\x00" in value
        or any(ord(character) < 0x20 for character in value)
    ):
        raise BuilderError(f"unsafe {description}: {value!r}")


def _parse_unsigned_decimal(value: str, description: str) -> int:
    if _UNSIGNED_DECIMAL_RE.fullmatch(value) is None:
        raise BuilderError(f"{description} is not an unsigned decimal integer")
    result = int(value)
    if result > (1 << 63) - 1:
        raise BuilderError(f"{description} is out of range")
    return result


def _parse_signed_crc32(value: str, description: str) -> int:
    if _DECIMAL_RE.fullmatch(value) is None:
        raise BuilderError(f"{description} is not a signed decimal integer")
    result = int(value)
    if not -(1 << 31) <= result <= (1 << 31) - 1:
        raise BuilderError(f"{description} is outside signed 32-bit range")
    return result


def _run_checked(
    command: Sequence[str],
    *,
    description: str,
    log_path: Path,
    pass_fds: Sequence[int] = (),
) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    try:
        log_fd = os.open(log_path, flags, 0o600)
    except OSError as exc:
        raise BuilderError(f"cannot create private command log: {exc}") from exc
    try:
        result = subprocess.run(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=log_fd,
            stderr=subprocess.STDOUT,
            pass_fds=tuple(pass_fds),
            close_fds=True,
            check=False,
        )
        os.fsync(log_fd)
    except OSError as exc:
        raise BuilderError(f"cannot {description}: {exc}") from exc
    finally:
        os.close(log_fd)
    if result.returncode != 0:
        detail = _read_log_tail(log_path)
        message = f"{description} failed with status {result.returncode}"
        if detail:
            message += f": {detail}"
        raise BuilderError(message)


def _run_to_private_files(
    command: Sequence[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    descriptors: list[int] = []
    try:
        for path in (stdout_path, stderr_path):
            descriptors.append(os.open(path, flags, 0o600))
        result = subprocess.run(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=descriptors[0],
            stderr=descriptors[1],
            close_fds=True,
            check=False,
        )
        for descriptor in descriptors:
            os.fsync(descriptor)
        return result.returncode
    except OSError as exc:
        raise BuilderError(f"cannot run host tool {command[0]}: {exc}") from exc
    finally:
        for descriptor in descriptors:
            os.close(descriptor)


def _read_log_tail(path: Path) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            stream.seek(max(0, size - COMMAND_LOG_TAIL))
            return _bounded_decode(stream.read(COMMAND_LOG_TAIL))
    except OSError:
        return ""


def _bounded_decode(value: bytes) -> str:
    return value.decode("utf-8", errors="replace").strip()


def _require_private_regular(
    path: Path,
    *,
    description: str,
    expected_size: int | None = None,
) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise BuilderError(f"cannot inspect {description}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise BuilderError(f"{description} is not a regular file")
    if expected_size is not None and metadata.st_size != expected_size:
        raise BuilderError(
            f"{description} is {metadata.st_size} bytes, "
            f"expected {expected_size}"
        )
    os.chmod(path, 0o600)


def _remove_private_tree(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        return
