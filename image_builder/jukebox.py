from __future__ import annotations

import hashlib
import os
import shutil
import stat
import struct
import subprocess
from pathlib import Path
from typing import Callable

from .errors import BuilderError
from .extfs import (
    ExtSuperblock,
    debugfs_cat,
    debugfs_require_path,
    parse_superblock,
    run_e2fsck_path,
)
from .io_utils import (
    OpenedRegular,
    copy_sparse_to_offsets,
    open_regular_read,
    pwrite_all,
)

Progress = Callable[[str], None]

JUKEBOX_TARGET_PATH = "/app/lib/libMediaPlayer.so.1.0.0"
JUKEBOX_TARGET_BYTES = 1_456_272
JUKEBOX_TARGET_SHA256 = (
    "3e80239bf2c698c1b2eaf63851976cee1141d9445a689376208633c5546be2a4"
)
JUKEBOX_HUNK_OFFSET = 875_906

# UpdateStorage() subtracts this value from statfs total bytes before comparing
# that result with available bytes.  The OEM value includes the stock p10
# reserved blocks.  Keep its non-reserved component and replace only the part
# that scales with the generated p10 filesystem.
OEM_P10_BLOCK_SIZE = 4096
OEM_P10_RESERVED_BLOCKS = 380_138
OEM_JUKEBOX_CORRECTION_BYTES = 1_737_420_800
OEM_NON_RESERVED_CORRECTION_BYTES = (
    OEM_JUKEBOX_CORRECTION_BYTES
    - OEM_P10_RESERVED_BLOCKS * OEM_P10_BLOCK_SIZE
)

_STOCK_CONTEXT = bytes.fromhex(
    "05001071988dbd64ffffff893424897c240483d2ff8945d88955dc"
)
_ADD_IMMEDIATE = slice(1, 5)
_ADC_IMMEDIATE = 20


def jukebox_correction_bytes(p10: ExtSuperblock) -> int:
    """Return the storage correction for the p10 filesystem actually built."""

    if p10.block_size != OEM_P10_BLOCK_SIZE:
        raise BuilderError(
            f"Jukebox correction requires a {OEM_P10_BLOCK_SIZE}-byte p10 "
            f"block size, got {p10.block_size}"
        )
    if not 0 <= p10.reserved_block_count <= p10.block_count:
        raise BuilderError("p10 has an invalid reserved-block count")
    correction = p10.reserved_bytes + OEM_NON_RESERVED_CORRECTION_BYTES
    if correction <= 0 or correction >= 1 << 63:
        raise BuilderError("Jukebox storage correction is outside the valid range")
    return correction


def jukebox_patch_bytes(p10: ExtSuperblock) -> tuple[int, bytes]:
    """Encode the dynamic 64-bit subtraction in UpdateStorage()."""

    correction = jukebox_correction_bytes(p10)
    signed_adjustment = -correction
    high_dword = signed_adjustment >> 32
    if not -128 <= high_dword <= 127:
        raise BuilderError(
            "Jukebox storage correction cannot be encoded by the OEM instruction"
        )
    replacement = bytearray(_STOCK_CONTEXT)
    replacement[_ADD_IMMEDIATE] = struct.pack(
        "<I", signed_adjustment & 0xFFFFFFFF
    )
    replacement[_ADC_IMMEDIATE] = high_dword & 0xFF
    return correction, bytes(replacement)


def apply_jukebox_fix(
    root_path: Path,
    p10: ExtSuperblock,
    *,
    progress: Progress | None = None,
) -> dict[str, object]:
    """Apply the geometry-specific Jukebox fix to a private root filesystem."""

    root_path = Path(root_path)
    correction, replacement = jukebox_patch_bytes(p10)
    if progress is not None:
        progress(
            "correcting Jukebox storage reporting for the generated p10 "
            f"({correction} bytes)"
        )

    with open_regular_read(root_path) as root:
        mapped_identity = (root.identity[0], root.identity[1], root.size)
        root_superblock = parse_superblock(root.fd)
        metadata = debugfs_require_path(
            root,
            JUKEBOX_TARGET_PATH,
            expected_type="regular",
            include_inode=True,
        )
        if metadata["size"] != JUKEBOX_TARGET_BYTES:
            raise BuilderError(
                f"Jukebox target size is {metadata['size']}, "
                f"expected {JUKEBOX_TARGET_BYTES}"
            )
        target_before = debugfs_cat(root, JUKEBOX_TARGET_PATH)
        if len(target_before) != JUKEBOX_TARGET_BYTES:
            raise BuilderError("Jukebox target changed while it was being read")
        target_sha256_before = hashlib.sha256(target_before).hexdigest()
        if target_sha256_before != JUKEBOX_TARGET_SHA256:
            raise BuilderError(
                "Jukebox target fingerprint does not match the supported "
                "application"
            )
        actual = target_before[
            JUKEBOX_HUNK_OFFSET : JUKEBOX_HUNK_OFFSET + len(_STOCK_CONTEXT)
        ]
        if actual != _STOCK_CONTEXT:
            if actual == replacement:
                raise BuilderError("Jukebox storage correction is already applied")
            raise BuilderError(
                "Jukebox source preimage does not match the supported application"
            )
        root_offset = _map_hunk(
            root,
            root_superblock.block_size,
            _STOCK_CONTEXT,
        )
        root.assert_unchanged()

    target_after = target_before
    if replacement != _STOCK_CONTEXT:
        writable = _open_writable_matching(root_path, mapped_identity)
        write_attempted = False
        try:
            try:
                if (
                    os.pread(writable.fd, len(_STOCK_CONTEXT), root_offset)
                    != _STOCK_CONTEXT
                ):
                    raise BuilderError(
                        "Jukebox mapped filesystem preimage changed before write"
                    )
                write_attempted = True
                pwrite_all(writable.fd, replacement, root_offset)
                os.fsync(writable.fd)
                if (
                    os.pread(writable.fd, len(replacement), root_offset)
                    != replacement
                ):
                    raise BuilderError("Jukebox verification read failed")
                target_after = debugfs_cat(writable, JUKEBOX_TARGET_PATH)
                if len(target_after) != len(target_before):
                    raise BuilderError("Jukebox target size changed")
                if (
                    target_after[
                        JUKEBOX_HUNK_OFFSET : JUKEBOX_HUNK_OFFSET
                        + len(replacement)
                    ]
                    != replacement
                ):
                    raise BuilderError("Jukebox target verification failed")
            except Exception as exc:
                if write_attempted:
                    try:
                        pwrite_all(writable.fd, _STOCK_CONTEXT, root_offset)
                        os.fsync(writable.fd)
                        if (
                            os.pread(
                                writable.fd,
                                len(_STOCK_CONTEXT),
                                root_offset,
                            )
                            != _STOCK_CONTEXT
                        ):
                            raise BuilderError("Jukebox rollback verification failed")
                    except Exception as rollback_exc:
                        raise BuilderError(
                            "Jukebox correction failed and rollback could not be "
                            f"verified: {rollback_exc}"
                        ) from exc
                raise
        finally:
            writable.close()

    if progress is not None:
        progress("checking the Jukebox-corrected root filesystem")
    run_e2fsck_path(root_path, "Jukebox-corrected restored root")
    changed_bytes = sum(
        before != after for before, after in zip(_STOCK_CONTEXT, replacement)
    )
    return {
        "schema": "genesis-jukebox-storage-fix/v1",
        "target": JUKEBOX_TARGET_PATH,
        "target_bytes": JUKEBOX_TARGET_BYTES,
        "target_inode": metadata["inode"],
        "file_offset": JUKEBOX_HUNK_OFFSET,
        "p10_block_size": p10.block_size,
        "p10_block_count": p10.block_count,
        "p10_reserved_block_count": p10.reserved_block_count,
        "correction_bytes": correction,
        "changed_bytes": changed_bytes,
        "sha256_before": target_sha256_before,
        "sha256_after": hashlib.sha256(target_after).hexdigest(),
    }


def stage_jukebox_fix(
    source: OpenedRegular,
    destination: Path,
    p10: ExtSuperblock,
    *,
    progress: Progress | None = None,
) -> tuple[OpenedRegular, dict[str, object]]:
    """Copy a root filesystem and apply the intrinsic fix to its private copy."""

    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise BuilderError(
            f"Jukebox staging destination already exists: {destination}"
        )
    source.assert_unchanged()
    if progress is not None:
        progress("staging a private sparse copy for the Jukebox correction")
    destination_fd = -1
    try:
        destination_fd = os.open(
            destination,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o600,
        )
        os.ftruncate(destination_fd, source.size)
        copy_sparse_to_offsets(source.fd, destination_fd, source.size, (0,))
        os.fsync(destination_fd)
    except Exception:
        if destination_fd >= 0:
            os.close(destination_fd)
            destination_fd = -1
        _unlink_stage(destination)
        raise
    finally:
        if destination_fd >= 0:
            os.close(destination_fd)

    try:
        report = apply_jukebox_fix(destination, p10, progress=progress)
        source.assert_unchanged()
        opened = open_regular_read(destination)
        if opened.size != source.size:
            opened.close()
            raise BuilderError("Jukebox-corrected root filesystem changed size")
        return opened, report
    except Exception:
        _unlink_stage(destination)
        raise


def _map_hunk(
    root: OpenedRegular,
    block_size: int,
    before: bytes,
) -> int:
    logical_block, in_block = divmod(JUKEBOX_HUNK_OFFSET, block_size)
    if in_block + len(before) > block_size:
        raise BuilderError("Jukebox hunk unexpectedly crosses a filesystem block")
    physical_block = _debugfs_bmap(root, logical_block)
    root_offset = physical_block * block_size + in_block
    if root_offset + len(before) > root.size:
        raise BuilderError("Jukebox target maps outside the root filesystem")
    return root_offset


def _debugfs_bmap(root: OpenedRegular, logical_block: int) -> int:
    tool = shutil.which("debugfs")
    if tool is None:
        raise BuilderError("required local tool is not installed: debugfs")
    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    environment["LANG"] = "C"
    try:
        result = subprocess.run(
            [
                tool,
                "-R",
                f"bmap {JUKEBOX_TARGET_PATH} {logical_block}",
                f"/proc/self/fd/{root.fd}",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=(root.fd,),
            check=False,
            timeout=120,
            env=environment,
        )
    except subprocess.TimeoutExpired as exc:
        raise BuilderError("debugfs timed out mapping the Jukebox target") from exc
    output = result.stdout.decode("ascii", errors="replace").strip()
    if result.returncode != 0 or not output.isdecimal():
        details = (result.stdout + result.stderr).decode(
            "utf-8", errors="replace"
        ).strip()
        raise BuilderError(
            "debugfs could not map the Jukebox target: " + details[-1000:]
        )
    physical_block = int(output)
    if physical_block <= 0:
        raise BuilderError("Jukebox target has a sparse hole at the patch offset")
    return physical_block


def _open_writable_matching(
    path: Path,
    expected_identity: tuple[int, int, int],
) -> OpenedRegular:
    flags = os.O_RDWR | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise BuilderError(
            f"cannot open Jukebox root stage {path}: {exc.strerror}"
        ) from exc
    try:
        current = os.fstat(fd)
        if not stat.S_ISREG(current.st_mode):
            raise BuilderError(f"Jukebox root stage is not a regular file: {path}")
        if (current.st_dev, current.st_ino, current.st_size) != expected_identity:
            raise BuilderError("Jukebox root stage changed before writing")
        return OpenedRegular(
            path=path,
            fd=fd,
            size=current.st_size,
            identity=(
                current.st_dev,
                current.st_ino,
                current.st_size,
                current.st_mtime_ns,
                current.st_ctime_ns,
            ),
        )
    except Exception:
        os.close(fd)
        raise


def _unlink_stage(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise BuilderError(
            f"cannot remove failed Jukebox root stage {path}: {exc.strerror}"
        ) from exc
