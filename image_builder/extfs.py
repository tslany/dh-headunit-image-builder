from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import struct
import subprocess
import uuid as uuid_module
from dataclasses import dataclass
from pathlib import Path

from .errors import BuilderError
from .geometry import PARTITION_BY_NUMBER, SYSLINUX_MBR_BYTES, SYSLINUX_MBR_SHA256
from .io_utils import OpenedRegular, open_regular_read

EXT_MAGIC = 0xEF53
EXT_SUPERBLOCK_OFFSET = 1024
EXT_SUPERBLOCK_SIZE = 1024
EXT_STATE_CLEAN = 0x0001
FEATURE_COMPAT_LEGACY = 0x3C
FEATURE_INCOMPAT_EXT4_LEGACY = 0x242
FEATURE_RO_COMPAT_EXT4_LEGACY = 0x7B
FEATURE_RO_COMPAT_P9 = 0x79

RESOURCE_DIRECTORY = Path(__file__).with_name("resources")
P7_SKELETON_PATH = RESOURCE_DIRECTORY / "p7_skeleton.json"
P8_SKELETON_PATH = RESOURCE_DIRECTORY / "p8_skeleton.json"
P10_SKELETON_PATH = RESOURCE_DIRECTORY / "p10_skeleton.json"
SKELETON_PATHS = {
    7: P7_SKELETON_PATH,
    8: P8_SKELETON_PATH,
    10: P10_SKELETON_PATH,
}
SKELETON_SCHEMAS = {
    7: "genesis-p7-skeleton/v1",
    8: "genesis-p8-skeleton/v1",
    10: "genesis-p10-skeleton/v1",
}
P7_SKELETON_SHA256 = hashlib.sha256(P7_SKELETON_PATH.read_bytes()).hexdigest()
SKELETON_SHA256 = {
    str(number): hashlib.sha256(path.read_bytes()).hexdigest()
    for number, path in SKELETON_PATHS.items()
}

_SAFE_PATH_COMPONENT = re.compile(r"[A-Za-z0-9._+-]+\Z")
_DEBUGFS_STAT = re.compile(
    r"Inode:\s+\d+\s+Type:\s+(\S+)\s+Mode:\s+([0-7]+).*?"
    r"User:\s+(\d+)\s+Group:\s+(\d+)",
    re.DOTALL,
)
_DEBUGFS_SIZE = re.compile(r"\bSize:\s+(\d+)\b")
_DEBUGFS_INODE = re.compile(r"\bInode:\s+(\d+)\b")
_DEBUGFS_LINK_TARGET = re.compile(r'Fast link dest:\s+"([^"]*)"', re.DOTALL)
_DEBUGFS_ALLOCATED_INODE = re.compile(rb"\bAllocated inode:\s*(\d+)\b")


@dataclass(frozen=True)
class ExtSuperblock:
    inode_count: int
    block_count: int
    reserved_block_count: int
    block_size: int
    blocks_per_group: int
    inodes_per_group: int
    inode_size: int
    uuid: str
    state: int
    feature_compat: int
    feature_incompat: int
    feature_ro_compat: int
    journal_bytes: int
    default_mount_options: int
    min_extra_isize: int
    want_extra_isize: int

    @property
    def filesystem_bytes(self) -> int:
        return self.block_count * self.block_size

    @property
    def reserved_bytes(self) -> int:
        return self.reserved_block_count * self.block_size

    def as_dict(self) -> dict[str, int | str | bool]:
        return {
            "inode_count": self.inode_count,
            "block_count": self.block_count,
            "reserved_block_count": self.reserved_block_count,
            "block_size": self.block_size,
            "blocks_per_group": self.blocks_per_group,
            "inodes_per_group": self.inodes_per_group,
            "inode_size": self.inode_size,
            "uuid": self.uuid,
            "clean": bool(self.state & EXT_STATE_CLEAN),
            "feature_compat": self.feature_compat,
            "feature_incompat": self.feature_incompat,
            "feature_ro_compat": self.feature_ro_compat,
            "journal_bytes": self.journal_bytes,
            "default_mount_options": self.default_mount_options,
            "min_extra_isize": self.min_extra_isize,
            "want_extra_isize": self.want_extra_isize,
        }


def parse_superblock(fd: int, base_offset: int = 0) -> ExtSuperblock:
    raw = os.pread(fd, EXT_SUPERBLOCK_SIZE, base_offset + EXT_SUPERBLOCK_OFFSET)
    if len(raw) != EXT_SUPERBLOCK_SIZE:
        raise BuilderError(f"short ext superblock at byte {base_offset + 1024}")
    magic = struct.unpack_from("<H", raw, 56)[0]
    if magic != EXT_MAGIC:
        raise BuilderError(
            f"no ext superblock at byte {base_offset + 1024}: magic 0x{magic:04x}"
        )
    inode_count, block_count, reserved_block_count = struct.unpack_from(
        "<III", raw, 0
    )
    log_block_size = struct.unpack_from("<I", raw, 24)[0]
    if log_block_size > 6:
        raise BuilderError(f"implausible ext block-size shift {log_block_size}")
    block_size = 1024 << log_block_size
    blocks_per_group = struct.unpack_from("<I", raw, 32)[0]
    inodes_per_group = struct.unpack_from("<I", raw, 40)[0]
    state = struct.unpack_from("<H", raw, 58)[0]
    inode_size = struct.unpack_from("<H", raw, 88)[0]
    feature_compat, feature_incompat, feature_ro_compat = struct.unpack_from(
        "<III", raw, 92
    )
    uuid_value = str(uuid_module.UUID(bytes=bytes(raw[104:120])))
    journal_bytes = _read_journal_bytes(
        fd,
        raw,
        base_offset=base_offset,
        block_size=block_size,
        inode_size=inode_size,
        inodes_per_group=inodes_per_group,
    )
    default_mount_options = struct.unpack_from("<I", raw, 256)[0]
    min_extra_isize, want_extra_isize = struct.unpack_from("<HH", raw, 348)
    return ExtSuperblock(
        inode_count=inode_count,
        block_count=block_count,
        reserved_block_count=reserved_block_count,
        block_size=block_size,
        blocks_per_group=blocks_per_group,
        inodes_per_group=inodes_per_group,
        inode_size=inode_size,
        uuid=uuid_value,
        state=state,
        feature_compat=feature_compat,
        feature_incompat=feature_incompat,
        feature_ro_compat=feature_ro_compat,
        journal_bytes=journal_bytes,
        default_mount_options=default_mount_options,
        min_extra_isize=min_extra_isize,
        want_extra_isize=want_extra_isize,
    )


def _read_journal_bytes(
    fd: int,
    superblock: bytes,
    *,
    base_offset: int,
    block_size: int,
    inode_size: int,
    inodes_per_group: int,
) -> int:
    journal_inode = struct.unpack_from("<I", superblock, 224)[0]
    if journal_inode == 0:
        return 0
    if inode_size < 128 or inodes_per_group <= 0:
        raise BuilderError("invalid ext geometry while locating journal inode")
    first_data_block = struct.unpack_from("<I", superblock, 20)[0]
    descriptor_size = struct.unpack_from("<H", superblock, 254)[0] or 32
    if descriptor_size < 32 or descriptor_size > block_size:
        raise BuilderError(f"implausible ext group-descriptor size {descriptor_size}")
    journal_group = (journal_inode - 1) // inodes_per_group
    journal_index = (journal_inode - 1) % inodes_per_group
    descriptor_offset = (
        base_offset
        + (first_data_block + 1) * block_size
        + journal_group * descriptor_size
    )
    descriptor = os.pread(fd, descriptor_size, descriptor_offset)
    if len(descriptor) != descriptor_size:
        raise BuilderError("short ext group descriptor while locating journal")
    inode_table_block = struct.unpack_from("<I", descriptor, 8)[0]
    inode_offset = (
        base_offset + inode_table_block * block_size + journal_index * inode_size
    )
    inode = os.pread(fd, inode_size, inode_offset)
    if len(inode) != inode_size:
        raise BuilderError("short ext journal inode")
    size_low = struct.unpack_from("<I", inode, 4)[0]
    size_high = struct.unpack_from("<I", inode, 108)[0]
    return size_low | (size_high << 32)


def validate_superblock(
    superblock: ExtSuperblock,
    spec: dict[str, object],
    *,
    description: str,
    container_bytes: int,
) -> None:
    checked_fields = (
        "inode_count",
        "block_count",
        "block_size",
        "blocks_per_group",
        "inodes_per_group",
        "inode_size",
        "uuid",
        "feature_compat",
        "feature_incompat",
        "feature_ro_compat",
        "journal_bytes",
        "default_mount_options",
        "min_extra_isize",
        "want_extra_isize",
    )
    expected = {field: spec[field] for field in checked_fields}
    actual = {field: getattr(superblock, field) for field in checked_fields}
    mismatches = [
        f"{key}={actual[key]!r} (expected {value!r})"
        for key, value in expected.items()
        if actual[key] != value
    ]
    if mismatches:
        raise BuilderError(f"{description} filesystem mismatch: " + "; ".join(mismatches))
    if superblock.filesystem_bytes > container_bytes:
        raise BuilderError(
            f"{description} filesystem is {superblock.filesystem_bytes} bytes but "
            f"container is {container_bytes}"
        )
    if not superblock.state & EXT_STATE_CLEAN:
        raise BuilderError(f"{description} filesystem is not marked clean")


def _require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise BuilderError(f"required local tool is not installed: {name}")
    return path


def debugfs_cat(opened: OpenedRegular, target_path: str) -> bytes:
    if not _safe_manifest_path(target_path):
        raise BuilderError(f"unsafe filesystem path: {target_path!r}")
    tool = _require_tool("debugfs")
    descriptor_path = f"/proc/self/fd/{opened.fd}"
    try:
        result = subprocess.run(
            [tool, "-R", f"cat {target_path}", descriptor_path],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=(opened.fd,),
            check=False,
            timeout=120,
            env=_tool_environment(),
        )
    except subprocess.TimeoutExpired as exc:
        raise BuilderError(f"debugfs timed out reading {target_path}") from exc
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise BuilderError(f"debugfs could not read {target_path}: {message}")
    lower_error = result.stderr.lower()
    if b"file not found" in lower_error or b"not found by ext2_lookup" in lower_error:
        raise BuilderError(f"filesystem path is missing: {target_path}")
    return result.stdout


def debugfs_require_path(
    opened: OpenedRegular,
    target_path: str,
    *,
    expected_type: str | None = None,
    include_inode: bool = False,
) -> dict[str, object]:
    if not _safe_manifest_path(target_path):
        raise BuilderError(f"unsafe filesystem path: {target_path!r}")
    tool = _require_tool("debugfs")
    descriptor_path = f"/proc/self/fd/{opened.fd}"
    try:
        result = subprocess.run(
            [tool, "-R", f"stat {target_path}", descriptor_path],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=(opened.fd,),
            check=False,
            timeout=120,
            env=_tool_environment(),
        )
    except subprocess.TimeoutExpired as exc:
        raise BuilderError(f"debugfs timed out inspecting {target_path}") from exc
    combined = result.stdout + result.stderr
    lowered = combined.lower()
    if (
        result.returncode != 0
        or b"file not found" in lowered
        or b"not found by ext2_lookup" in lowered
    ):
        output = combined.decode("utf-8", errors="replace").strip()
        raise BuilderError(
            f"required filesystem path is missing: {target_path}: {output[-1000:]}"
        )
    text = result.stdout.decode("utf-8", errors="replace")
    match = _DEBUGFS_STAT.search(text)
    if match is None:
        raise BuilderError(f"cannot parse filesystem metadata for {target_path}")
    inode_type, mode_text, uid_text, gid_text = match.groups()
    size_match = _DEBUGFS_SIZE.search(text)
    if size_match is None:
        raise BuilderError(f"cannot parse filesystem size for {target_path}")
    if expected_type is not None and inode_type != expected_type:
        raise BuilderError(
            f"filesystem path {target_path} is {inode_type}, expected {expected_type}"
        )
    link_match = _DEBUGFS_LINK_TARGET.search(text)
    metadata: dict[str, object] = {
        "path": target_path,
        "type": inode_type,
        "mode": int(mode_text, 8),
        "uid": int(uid_text),
        "gid": int(gid_text),
        "size": int(size_match.group(1)),
        "link_target": link_match.group(1) if link_match is not None else None,
    }
    if include_inode:
        inode_match = _DEBUGFS_INODE.search(text)
        if inode_match is None:
            raise BuilderError(f"cannot parse filesystem inode for {target_path}")
        metadata["inode"] = int(inode_match.group(1))
    return metadata


def debugfs_require_absent(opened: OpenedRegular, target_path: str) -> None:
    """Require one safe filesystem path to be absent."""
    if not _safe_manifest_path(target_path):
        raise BuilderError(f"unsafe filesystem path: {target_path!r}")
    tool = _require_tool("debugfs")
    descriptor_path = f"/proc/self/fd/{opened.fd}"
    try:
        result = subprocess.run(
            [tool, "-R", f"stat {target_path}", descriptor_path],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=(opened.fd,),
            check=False,
            timeout=120,
            env=_tool_environment(),
        )
    except subprocess.TimeoutExpired as exc:
        raise BuilderError(f"debugfs timed out inspecting {target_path}") from exc
    combined = result.stdout + result.stderr
    lowered = combined.lower()
    text = result.stdout.decode("utf-8", errors="replace")
    if _DEBUGFS_STAT.search(text) is not None:
        raise BuilderError(f"filesystem path already exists: {target_path}")
    if b"file not found" in lowered or b"not found by ext2_lookup" in lowered:
        return
    if result.returncode != 0:
        output = combined.decode("utf-8", errors="replace").strip()
        raise BuilderError(
            f"debugfs could not inspect {target_path}: {output[-1000:]}"
        )
    raise BuilderError(
        f"cannot determine whether filesystem path exists: {target_path}"
    )


def debugfs_create_regular(
    filesystem_path: Path,
    target_path: str,
    payload_fd: int,
    *,
    mode: int,
    uid: int,
    gid: int,
) -> None:
    """Create one absent regular file without mounting the filesystem."""
    if not _safe_manifest_path(target_path):
        raise BuilderError(f"unsafe filesystem path: {target_path!r}")
    if (
        type(mode) is not int
        or mode < 0
        or mode > 0o777
        or type(uid) is not int
        or uid < 0
        or uid > 0x7FFFFFFF
        or type(gid) is not int
        or gid < 0
        or gid > 0x7FFFFFFF
    ):
        raise BuilderError("invalid created file metadata")
    payload_identity, payload_size, payload_sha256 = _regular_fd_snapshot(
        payload_fd,
        "creation payload",
    )

    with open_regular_read(filesystem_path) as root:
        current = ""
        for component in target_path.split("/")[1:-1]:
            current += "/" + component
            debugfs_require_path(root, current, expected_type="directory")
        debugfs_require_absent(root, target_path)
        root.assert_unchanged()

    write_output = _run_debugfs_batch(
        filesystem_path,
        [f"write /proc/self/fd/{payload_fd} {target_path}"],
        f"create {target_path}",
        pass_fds=(payload_fd,),
        capture_output=True,
    )
    assert write_output is not None
    allocated = _DEBUGFS_ALLOCATED_INODE.search(write_output)
    if allocated is None:
        output = write_output.decode("utf-8", errors="replace").strip()
        raise BuilderError(
            f"debugfs did not confirm creation of {target_path}: {output[-1000:]}"
        )
    created_inode = int(allocated.group(1))
    _require_fd_snapshot(
        payload_fd,
        payload_identity,
        payload_size,
        payload_sha256,
        "creation payload",
    )
    _verify_created_regular(
        filesystem_path,
        target_path,
        created_inode=created_inode,
        expected_size=payload_size,
        expected_sha256=payload_sha256,
    )

    complete_mode = 0o100000 | mode
    inode_reference = f"<{created_inode}>"
    _run_debugfs_batch(
        filesystem_path,
        [
            f"set_inode_field {inode_reference} mode 0{complete_mode:o}",
            f"set_inode_field {inode_reference} uid {uid}",
            f"set_inode_field {inode_reference} gid {gid}",
        ],
        f"set metadata for {target_path}",
    )
    _require_fd_snapshot(
        payload_fd,
        payload_identity,
        payload_size,
        payload_sha256,
        "creation payload",
    )
    metadata = _verify_created_regular(
        filesystem_path,
        target_path,
        created_inode=created_inode,
        expected_size=payload_size,
        expected_sha256=payload_sha256,
    )
    if (
        metadata["mode"] != mode
        or metadata["uid"] != uid
        or metadata["gid"] != gid
    ):
        raise BuilderError(
            f"created file metadata verification failed for {target_path}"
        )


def _regular_fd_snapshot(
    descriptor: int,
    description: str,
) -> tuple[tuple[int, int, int, int, int], int, str]:
    try:
        before = os.fstat(descriptor)
    except OSError as exc:
        raise BuilderError(f"cannot inspect {description}: {exc}") from exc
    if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
        raise BuilderError(f"{description} must be a nonempty regular file")
    digest = hashlib.sha256()
    position = 0
    while position < before.st_size:
        try:
            chunk = os.pread(
                descriptor,
                min(1024 * 1024, before.st_size - position),
                position,
            )
        except OSError as exc:
            raise BuilderError(f"cannot read {description}: {exc}") from exc
        if not chunk:
            raise BuilderError(f"unexpected EOF while reading {description}")
        digest.update(chunk)
        position += len(chunk)
    try:
        after = os.fstat(descriptor)
    except OSError as exc:
        raise BuilderError(f"cannot reinspect {description}: {exc}") from exc
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if after_identity != before_identity:
        raise BuilderError(f"{description} changed while being read")
    return before_identity, before.st_size, digest.hexdigest()


def _require_fd_snapshot(
    descriptor: int,
    expected_identity: tuple[int, int, int, int, int],
    expected_size: int,
    expected_sha256: str,
    description: str,
) -> None:
    identity, size, digest = _regular_fd_snapshot(descriptor, description)
    if (
        identity != expected_identity
        or size != expected_size
        or digest != expected_sha256
    ):
        raise BuilderError(f"{description} changed while in use")


def _verify_created_regular(
    filesystem_path: Path,
    target_path: str,
    *,
    created_inode: int,
    expected_size: int,
    expected_sha256: str,
) -> dict[str, object]:
    with open_regular_read(filesystem_path) as root:
        metadata = debugfs_require_path(
            root,
            target_path,
            expected_type="regular",
            include_inode=True,
        )
        data = debugfs_cat(root, target_path)
        root.assert_unchanged()
    if (
        metadata.get("inode") != created_inode
        or metadata.get("size") != expected_size
        or len(data) != expected_size
        or hashlib.sha256(data).hexdigest() != expected_sha256
    ):
        raise BuilderError(f"created file verification failed for {target_path}")
    return metadata


def debugfs_replace_regular(
    filesystem_path: Path,
    target_path: str,
    payload_fd: int,
    *,
    mode: int,
    uid: int,
    gid: int,
) -> None:
    """Replace one existing regular file without mounting the filesystem."""
    if not _safe_manifest_path(target_path):
        raise BuilderError(f"unsafe filesystem path: {target_path!r}")
    if (
        type(mode) is not int
        or mode < 0
        or mode > 0o777
        or type(uid) is not int
        or uid < 0
        or uid > 0xFFFFFFFF
        or type(gid) is not int
        or gid < 0
        or gid > 0xFFFFFFFF
    ):
        raise BuilderError("invalid replacement file metadata")
    try:
        payload_stat = os.fstat(payload_fd)
    except OSError as exc:
        raise BuilderError(f"cannot inspect replacement payload: {exc}") from exc
    if not stat.S_ISREG(payload_stat.st_mode) or payload_stat.st_size <= 0:
        raise BuilderError("replacement payload must be a nonempty regular file")

    complete_mode = 0o100000 | mode
    commands = [
        f"rm {target_path}",
        f"write /proc/self/fd/{payload_fd} {target_path}",
        f"set_inode_field {target_path} mode 0{complete_mode:o}",
        f"set_inode_field {target_path} uid {uid}",
        f"set_inode_field {target_path} gid {gid}",
    ]
    _run_debugfs_batch(
        filesystem_path,
        commands,
        f"replace {target_path}",
        pass_fds=(payload_fd,),
    )


def extract_syslinux_mbr(root: OpenedRegular) -> bytes:
    metadata = debugfs_require_path(
        root,
        "/usr/share/syslinux/mbr.bin",
        expected_type="regular",
    )
    if metadata["size"] != SYSLINUX_MBR_BYTES:
        raise BuilderError(
            "supplied root has an unsupported /usr/share/syslinux/mbr.bin "
            f"size: {metadata['size']}"
        )
    bootstrap = debugfs_cat(root, "/usr/share/syslinux/mbr.bin")
    digest = hashlib.sha256(bootstrap).hexdigest()
    if len(bootstrap) != SYSLINUX_MBR_BYTES or digest != SYSLINUX_MBR_SHA256:
        raise BuilderError(
            "supplied root has an unsupported /usr/share/syslinux/mbr.bin: "
            f"size={len(bootstrap)}, sha256={digest}"
        )
    return bootstrap


def run_e2fsck(opened: OpenedRegular, description: str) -> None:
    tool = _require_tool("e2fsck")
    descriptor_path = f"/proc/self/fd/{opened.fd}"
    try:
        result = subprocess.run(
            [tool, "-fn", descriptor_path],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            pass_fds=(opened.fd,),
            check=False,
            timeout=600,
            env=_tool_environment(),
        )
    except subprocess.TimeoutExpired as exc:
        raise BuilderError(f"e2fsck timed out for {description}") from exc
    if result.returncode != 0:
        output = result.stdout.decode("utf-8", errors="replace")
        raise BuilderError(
            f"non-writing e2fsck failed for {description} with exit "
            f"{result.returncode}:\n{output[-4000:]}"
        )


def run_e2fsck_path(path: Path, description: str) -> None:
    tool = _require_tool("e2fsck")
    try:
        result = subprocess.run(
            [tool, "-fn", os.fspath(path)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=600,
            env=_tool_environment(),
        )
    except subprocess.TimeoutExpired as exc:
        raise BuilderError(f"e2fsck timed out for {description}") from exc
    if result.returncode != 0:
        output = result.stdout.decode("utf-8", errors="replace")
        raise BuilderError(
            f"non-writing e2fsck failed for {description} with exit "
            f"{result.returncode}:\n{output[-4000:]}"
        )


def fresh_ext_spec(partition_number: int) -> dict[str, int]:
    p10_block_size = 4096
    p10_blocks_per_group = 32_768
    p10_inodes_per_group = 8_160
    p10_block_count = PARTITION_BY_NUMBER[10].bytes // p10_block_size
    p10_group_count = (
        p10_block_count + p10_blocks_per_group - 1
    ) // p10_blocks_per_group
    canonical_geometry = {
        7: {
            "block_count": 393_216,
            "inode_count": 98_496,
            "inodes_per_group": 8_208,
            "journal_size_mb": 32,
            "root_uid": 0,
            "root_gid": 0,
        },
        8: {
            "block_count": 263_064,
            "inode_count": 65_808,
            "inodes_per_group": 7_312,
            "journal_size_mb": 32,
            "root_uid": 500,
            "root_gid": 500,
        },
        9: {
            "block_count": 104_420,
            "inode_count": 26_208,
            "inodes_per_group": 2_016,
            "journal_size_mb": 4,
            "root_uid": 500,
            "root_gid": 500,
        },
        10: {
            "block_count": p10_block_count,
            "inode_count": p10_group_count * p10_inodes_per_group,
            "inodes_per_group": p10_inodes_per_group,
            "journal_size_mb": 128,
            "root_uid": 500,
            "root_gid": 500,
        },
        12: {
            "block_count": 1_050_368,
            "inode_count": 262_944,
            "inodes_per_group": 7_968,
            "journal_size_mb": 128,
            "root_uid": 500,
            "root_gid": 500,
        },
        13: {
            "block_count": 1_311_488,
            "inode_count": 328_000,
            "inodes_per_group": 8_000,
            "journal_size_mb": 128,
            "root_uid": 500,
            "root_gid": 500,
        },
    }
    if partition_number not in canonical_geometry:
        raise BuilderError(f"no fresh-filesystem specification for p{partition_number}")
    if partition_number == 9:
        common = {
            "block_size": 1024,
            "blocks_per_group": 8192,
            "inode_size": 128,
            "feature_compat": FEATURE_COMPAT_LEGACY,
            "feature_incompat": FEATURE_INCOMPAT_EXT4_LEGACY,
            "feature_ro_compat": FEATURE_RO_COMPAT_P9,
            "default_mount_options": 0,
            "min_extra_isize": 0,
            "want_extra_isize": 0,
        }
    else:
        common = {
            "block_size": 4096,
            "blocks_per_group": 32768,
            "inode_size": 256,
            "feature_compat": FEATURE_COMPAT_LEGACY,
            "feature_incompat": FEATURE_INCOMPAT_EXT4_LEGACY,
            "feature_ro_compat": FEATURE_RO_COMPAT_EXT4_LEGACY,
            "default_mount_options": 0,
            "min_extra_isize": 28,
            "want_extra_isize": 28,
        }
    return {**common, **canonical_geometry[partition_number]}


def create_legacy_ext(path: Path, size: int, partition_number: int) -> ExtSuperblock:
    tool = _require_tool("mkfs.ext4")
    spec = fresh_ext_spec(partition_number)
    canonical_size = PARTITION_BY_NUMBER[partition_number].bytes
    canonical_geometry = size == canonical_size
    if size <= 0:
        raise BuilderError(f"p{partition_number} filesystem size must be positive")
    if canonical_geometry:
        block_count = int(spec["block_count"])
    else:
        block_count = size // int(spec["block_size"])
    if block_count <= 0:
        raise BuilderError(f"p{partition_number} filesystem has no complete blocks")
    if block_count * int(spec["block_size"]) > size:
        raise BuilderError(f"p{partition_number} filesystem exceeds its container")
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
    try:
        os.ftruncate(fd, size)
    finally:
        os.close(fd)
    features = [
        "none",
        "has_journal",
        "ext_attr",
        "resize_inode",
        "dir_index",
        "filetype",
        "extent",
        "flex_bg",
        "sparse_super",
        "huge_file",
        "uninit_bg",
        "dir_nlink",
        "extra_isize",
    ]
    if partition_number == 9:
        features.insert(8, "^large_file")
    else:
        features.insert(8, "large_file")
    filesystem_uuid = str(uuid_module.uuid4())
    command = [
        tool,
        "-q",
        "-F",
        "-b",
        str(spec["block_size"]),
        "-I",
        str(spec["inode_size"]),
        "-m",
        "5",
        "-J",
        f"size={spec['journal_size_mb']}",
        "-G",
        "16",
        "-g",
        str(spec["blocks_per_group"]),
        "-U",
        filesystem_uuid,
        "-O",
        ",".join(features),
        "-E",
        (
            "nodiscard,lazy_itable_init=0,lazy_journal_init=0,"
            f"root_owner={spec['root_uid']}:{spec['root_gid']},root_perms=0755"
        ),
    ]
    if canonical_geometry:
        command.extend(["-N", str(spec["inode_count"])])
    command.extend([os.fspath(path), str(block_count)])
    result = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=1200,
        env=_tool_environment(),
    )
    if result.returncode != 0:
        output = (result.stdout + result.stderr).decode("utf-8", errors="replace")
        raise BuilderError(
            f"mkfs.ext4 failed for p{partition_number} with exit "
            f"{result.returncode}:\n{output[-4000:]}"
        )
    _normalize_fresh_filesystem(path, partition_number, spec)
    if partition_number in (8, 10):
        populate_fresh_filesystem(path, partition_number)
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        superblock = parse_superblock(fd)
    finally:
        os.close(fd)
    if superblock.block_count != block_count:
        raise BuilderError(
            f"new p{partition_number} has block_count={superblock.block_count}, "
            f"expected {block_count}"
        )
    expected_reserved_blocks = block_count * 5 // 100
    if superblock.reserved_block_count != expected_reserved_blocks:
        raise BuilderError(
            f"new p{partition_number} has "
            f"reserved_block_count={superblock.reserved_block_count}, "
            f"expected {expected_reserved_blocks}"
        )
    for key in (
        "block_size",
        "blocks_per_group",
        "inode_size",
        "feature_compat",
        "feature_incompat",
        "feature_ro_compat",
        "default_mount_options",
        "min_extra_isize",
        "want_extra_isize",
    ):
        if getattr(superblock, key) != spec[key]:
            raise BuilderError(
                f"new p{partition_number} has {key}={getattr(superblock, key)!r}, "
                f"expected {spec[key]!r}"
            )
    if canonical_geometry and superblock.inode_count != spec["inode_count"]:
        raise BuilderError(
            f"new p{partition_number} has inode_count={superblock.inode_count}, "
            f"expected {spec['inode_count']}"
        )
    if canonical_geometry and superblock.inodes_per_group != spec["inodes_per_group"]:
        raise BuilderError(
            f"new p{partition_number} has "
            f"inodes_per_group={superblock.inodes_per_group}, "
            f"expected {spec['inodes_per_group']}"
        )
    if superblock.uuid != filesystem_uuid:
        raise BuilderError(f"new p{partition_number} UUID does not match requested UUID")
    expected_journal_bytes = int(spec["journal_size_mb"]) * 1024 * 1024
    if superblock.journal_bytes != expected_journal_bytes:
        raise BuilderError(
            f"new p{partition_number} has journal_bytes={superblock.journal_bytes}, "
            f"expected {expected_journal_bytes}"
        )
    if superblock.filesystem_bytes > size:
        raise BuilderError(f"new p{partition_number} filesystem exceeds its partition")
    return superblock


def _normalize_fresh_filesystem(
    path: Path, partition_number: int, spec: dict[str, int]
) -> None:
    root_uid = int(spec["root_uid"])
    root_gid = int(spec["root_gid"])
    commands = [
        f"set_super_value default_mount_opts {spec['default_mount_options']}",
        f"set_super_value min_extra_isize {spec['min_extra_isize']}",
        f"set_super_value want_extra_isize {spec['want_extra_isize']}",
        "set_inode_field / mode 040755",
        f"set_inode_field / uid {root_uid}",
        f"set_inode_field / gid {root_gid}",
        "set_inode_field /lost+found mode 040700",
        "set_inode_field /lost+found uid 500",
        "set_inode_field /lost+found gid 500",
    ]
    _run_debugfs_batch(path, commands, f"normalize p{partition_number}")
    _audit_inode(
        path,
        "/",
        expected_type="directory",
        expected_mode=0o755,
        expected_uid=root_uid,
        expected_gid=root_gid,
    )
    _audit_inode(
        path,
        "/lost+found",
        expected_type="directory",
        expected_mode=0o700,
        expected_uid=500,
        expected_gid=500,
    )


def populate_fresh_filesystem(
    path: Path,
    partition_number: int,
    *,
    manifest_path: Path | None = None,
) -> dict[str, object]:
    selected_manifest = (
        manifest_path
        if manifest_path is not None
        else SKELETON_PATHS.get(partition_number)
    )
    if selected_manifest is None:
        raise BuilderError(f"p{partition_number} has no filesystem skeleton")
    raw_manifest = selected_manifest.read_bytes()
    manifest_hash = hashlib.sha256(raw_manifest).hexdigest()
    if manifest_path is None:
        expected_hash = SKELETON_SHA256.get(str(partition_number))
        if expected_hash is None or manifest_hash != expected_hash:
            raise BuilderError(
                f"p{partition_number} skeleton changed while in use"
            )
    try:
        manifest = json.loads(raw_manifest)
    except json.JSONDecodeError as exc:
        raise BuilderError(
            f"invalid p{partition_number} skeleton manifest: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise BuilderError(f"p{partition_number} skeleton manifest is not an object")
    _validate_skeleton_manifest(manifest, partition_number)
    directories = manifest["directories"]
    symlinks = manifest["symlinks"]
    assert isinstance(directories, list)
    assert isinstance(symlinks, list)
    commands: list[str] = []
    for entry in directories:
        assert isinstance(entry, dict)
        path_value = entry["path"]
        mode = entry["mode"]
        uid = entry["uid"]
        gid = entry["gid"]
        commands.append(f"mkdir {path_value}")
        complete_mode = 0o040000 | int(mode, 8)
        commands.append(f"set_inode_field {path_value} mode 0{complete_mode:o}")
        commands.append(f"set_inode_field {path_value} uid {uid}")
        commands.append(f"set_inode_field {path_value} gid {gid}")
    for entry in symlinks:
        assert isinstance(entry, dict)
        path_value = entry["path"]
        commands.append(f"symlink {path_value} {entry['target']}")
        commands.append(f"set_inode_field {path_value} uid {entry['uid']}")
        commands.append(f"set_inode_field {path_value} gid {entry['gid']}")
    _run_debugfs_batch(path, commands, f"populate p{partition_number}")
    audit = audit_fresh_filesystem(
        path,
        partition_number,
        manifest_path=selected_manifest,
        manifest=manifest,
    )
    return {
        "partition": partition_number,
        "manifest": selected_manifest.name,
        "manifest_sha256": manifest_hash,
        "directories": len(directories),
        "symlinks": len(symlinks),
        "audited": audit["valid"],
    }


def populate_p7(path: Path) -> str:
    """Compatibility wrapper used by the existing image builder."""
    report = populate_fresh_filesystem(path, 7)
    manifest_hash = report["manifest_sha256"]
    assert isinstance(manifest_hash, str)
    return manifest_hash


def audit_fresh_filesystem(
    path: Path,
    partition_number: int,
    *,
    manifest_path: Path | None = None,
    manifest: dict[str, object] | None = None,
) -> dict[str, object]:
    selected_manifest = (
        manifest_path
        if manifest_path is not None
        else SKELETON_PATHS.get(partition_number)
    )
    if selected_manifest is None:
        raise BuilderError(f"p{partition_number} has no filesystem skeleton")
    if manifest is None:
        try:
            loaded = json.loads(selected_manifest.read_bytes())
        except json.JSONDecodeError as exc:
            raise BuilderError(
                f"invalid p{partition_number} skeleton manifest: {exc}"
            ) from exc
        if not isinstance(loaded, dict):
            raise BuilderError(
                f"p{partition_number} skeleton manifest is not an object"
            )
        manifest = loaded
    _validate_skeleton_manifest(manifest, partition_number)
    directories = manifest["directories"]
    symlinks = manifest["symlinks"]
    assert isinstance(directories, list)
    assert isinstance(symlinks, list)
    for entry in directories:
        assert isinstance(entry, dict)
        _audit_inode(
            path,
            entry["path"],
            expected_type="directory",
            expected_mode=int(entry["mode"], 8),
            expected_uid=entry["uid"],
            expected_gid=entry["gid"],
        )
    for entry in symlinks:
        assert isinstance(entry, dict)
        _audit_inode(
            path,
            entry["path"],
            expected_type="symlink",
            expected_mode=0o777,
            expected_uid=entry["uid"],
            expected_gid=entry["gid"],
            expected_target=entry["target"],
        )
    return {
        "valid": True,
        "partition": partition_number,
        "directories": len(directories),
        "symlinks": len(symlinks),
    }


def _validate_skeleton_manifest(
    manifest: dict[str, object], partition_number: int
) -> None:
    if set(manifest) != {"schema", "directories", "symlinks"}:
        raise BuilderError(
            f"p{partition_number} skeleton has invalid top-level fields"
        )
    expected_schema = SKELETON_SCHEMAS.get(partition_number)
    if expected_schema is None or manifest.get("schema") != expected_schema:
        raise BuilderError(f"unsupported p{partition_number} skeleton schema")
    directories = manifest.get("directories")
    symlinks = manifest.get("symlinks")
    if not isinstance(directories, list) or not isinstance(symlinks, list):
        raise BuilderError(f"p{partition_number} skeleton has invalid entry lists")
    seen = {"/", "/lost+found"}
    for entry in directories:
        if not isinstance(entry, dict):
            raise BuilderError(
                f"p{partition_number} skeleton directory entry is not an object"
            )
        if set(entry) != {"path", "mode", "uid", "gid"}:
            raise BuilderError(
                f"p{partition_number} directory entry has invalid fields: {entry!r}"
            )
        path_value = entry.get("path")
        if not isinstance(path_value, str) or not _safe_manifest_path(path_value):
            raise BuilderError(
                f"p{partition_number} directory path is invalid: {path_value!r}"
            )
        if path_value in seen:
            raise BuilderError(
                f"duplicate p{partition_number} skeleton path: {path_value}"
            )
        parent = os.fspath(Path(path_value).parent)
        if parent not in seen:
            raise BuilderError(
                f"p{partition_number} parent must precede child: {path_value}"
            )
        _validate_numeric_identity(entry, path_value, partition_number)
        mode = entry.get("mode")
        if not isinstance(mode, str) or re.fullmatch(r"[0-7]{4}", mode) is None:
            raise BuilderError(
                f"p{partition_number} mode is invalid: {path_value}"
            )
        seen.add(path_value)
    for entry in symlinks:
        if not isinstance(entry, dict):
            raise BuilderError(
                f"p{partition_number} skeleton symlink entry is not an object"
            )
        if set(entry) != {"path", "target", "uid", "gid"}:
            raise BuilderError(
                f"p{partition_number} symlink entry has invalid fields: {entry!r}"
            )
        path_value = entry.get("path")
        target = entry.get("target")
        if (
            not isinstance(path_value, str)
            or not _safe_manifest_path(path_value)
            or path_value in seen
            or not isinstance(target, str)
            or not _safe_symlink_target(target)
        ):
            raise BuilderError(
                f"invalid p{partition_number} symlink entry: {entry!r}"
            )
        parent = os.fspath(Path(path_value).parent)
        if parent not in seen:
            raise BuilderError(
                f"p{partition_number} symlink parent is missing: {path_value}"
            )
        _validate_numeric_identity(entry, path_value, partition_number)
        seen.add(path_value)


def _validate_numeric_identity(
    entry: dict[str, object], path_value: str, partition_number: int
) -> None:
    for field in ("uid", "gid"):
        value = entry.get(field)
        if type(value) is not int or not 0 <= value <= 0xFFFFFFFF:
            raise BuilderError(
                f"p{partition_number} {field} is invalid for {path_value}"
            )


def _safe_manifest_path(path_value: str) -> bool:
    if (
        not path_value.startswith("/")
        or path_value == "/"
        or path_value.endswith("/")
        or len(path_value.encode("utf-8")) > 4095
        or _has_control_character(path_value)
    ):
        return False
    components = path_value.split("/")[1:]
    return bool(components) and all(
        component not in ("", ".", "..")
        and _SAFE_PATH_COMPONENT.fullmatch(component) is not None
        for component in components
    )


def _safe_symlink_target(target: str) -> bool:
    if (
        not target
        or target.endswith("/")
        or len(target.encode("utf-8")) > 4095
        or _has_control_character(target)
    ):
        return False
    components = target.split("/")
    if target.startswith("/"):
        components = components[1:]
    return bool(components) and all(
        component not in ("", ".", "..")
        and _SAFE_PATH_COMPONENT.fullmatch(component) is not None
        for component in components
    )


def _has_control_character(value: str) -> bool:
    return any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)


def _run_debugfs_batch(
    path: Path,
    commands: list[str],
    description: str,
    *,
    pass_fds: tuple[int, ...] = (),
    capture_output: bool = False,
) -> bytes | None:
    tool = _require_tool("debugfs")
    command_path = path.with_suffix(path.suffix + ".debugfs-commands")
    encoded = ("\n".join(commands) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        command_fd = os.open(command_path, flags, 0o600)
    except OSError as exc:
        raise BuilderError(
            f"cannot create private debugfs command file for {description}: {exc}"
        ) from exc
    try:
        offset = 0
        while offset < len(encoded):
            written = os.write(command_fd, encoded[offset:])
            if written <= 0:
                raise BuilderError(
                    f"short write creating debugfs commands for {description}"
                )
            offset += written
    finally:
        os.close(command_fd)
    try:
        result = subprocess.run(
            [tool, "-w", "-f", os.fspath(command_path), os.fspath(path)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=pass_fds,
            check=False,
            timeout=300,
            env=_tool_environment(),
        )
    finally:
        command_path.unlink(missing_ok=True)
    if result.returncode != 0:
        output = (result.stdout + result.stderr).decode("utf-8", errors="replace")
        raise BuilderError(
            f"debugfs could not {description} with exit {result.returncode}:\n"
            f"{output[-4000:]}"
        )
    if capture_output:
        return result.stdout + result.stderr
    return None


def _audit_inode(
    path: Path,
    target_path: object,
    *,
    expected_type: str,
    expected_mode: int,
    expected_uid: object,
    expected_gid: object,
    expected_target: object | None = None,
) -> None:
    if not isinstance(target_path, str):
        raise BuilderError("internal skeleton audit path is not a string")
    result = _run_debugfs_read(path, f"stat {target_path}")
    match = _DEBUGFS_STAT.search(result)
    if match is None:
        raise BuilderError(f"cannot parse debugfs metadata for {target_path}")
    inode_type, mode_text, uid_text, gid_text = match.groups()
    actual = {
        "type": inode_type,
        "mode": int(mode_text, 8),
        "uid": int(uid_text),
        "gid": int(gid_text),
    }
    expected = {
        "type": expected_type,
        "mode": expected_mode,
        "uid": expected_uid,
        "gid": expected_gid,
    }
    mismatches = [
        f"{field}={actual[field]!r} (expected {value!r})"
        for field, value in expected.items()
        if actual[field] != value
    ]
    if expected_target is not None:
        target_match = _DEBUGFS_LINK_TARGET.search(result)
        actual_target = target_match.group(1) if target_match is not None else None
        if actual_target != expected_target:
            mismatches.append(
                f"target={actual_target!r} (expected {expected_target!r})"
            )
    if mismatches:
        raise BuilderError(
            f"filesystem skeleton audit failed for {target_path}: "
            + "; ".join(mismatches)
        )


def _run_debugfs_read(path: Path, request: str) -> str:
    tool = _require_tool("debugfs")
    try:
        result = subprocess.run(
            [tool, "-R", request, os.fspath(path)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=120,
            env=_tool_environment(),
        )
    except subprocess.TimeoutExpired as exc:
        raise BuilderError(f"debugfs timed out running {request!r}") from exc
    combined = result.stdout + result.stderr
    lowered = combined.lower()
    if (
        result.returncode != 0
        or b"file not found" in lowered
        or b"not found by ext2_lookup" in lowered
    ):
        output = combined.decode("utf-8", errors="replace").strip()
        raise BuilderError(f"debugfs failed running {request!r}: {output[-4000:]}")
    return result.stdout.decode("utf-8", errors="replace")


def _tool_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for variable in (
        "MKE2FS_CONFIG",
        "MKE2FS_FIRST_META_BG",
        "E2FSPROGS_FAKE_TIME",
        "DEBUGFS_PAGER",
        "PAGER",
    ):
        environment.pop(variable, None)
    environment["LC_ALL"] = "C"
    environment["LANG"] = "C"
    return environment


def create_swap(path: Path, size: int) -> dict[str, object]:
    tool = _require_tool("mkswap")
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
    try:
        os.ftruncate(fd, size)
    finally:
        os.close(fd)
    swap_uuid = str(uuid_module.uuid4())
    result = subprocess.run(
        [
            tool,
            "--pagesize",
            "4096",
            "--swapversion",
            "1",
            "--uuid",
            swap_uuid,
            os.fspath(path),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=120,
        env=_tool_environment(),
    )
    if result.returncode != 0:
        output = (result.stdout + result.stderr).decode("utf-8", errors="replace")
        raise BuilderError(f"mkswap failed with exit {result.returncode}:\n{output}")
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        header = os.pread(fd, 4096, 0)
    finally:
        os.close(fd)
    if len(header) != 4096:
        raise BuilderError("new swap partition has a short first page")
    signature = header[4096 - 10 :]
    if signature != b"SWAPSPACE2":
        raise BuilderError("new swap partition has no SWAPSPACE2 signature")
    version, last_page, bad_pages = struct.unpack_from("<III", header, 1024)
    observed_uuid = str(uuid_module.UUID(bytes=bytes(header[1036:1052])))
    expected_last_page = size // 4096 - 1
    if (
        version != 1
        or last_page != expected_last_page
        or bad_pages != 0
        or observed_uuid != swap_uuid
    ):
        raise BuilderError("new swap header does not match requested v1 geometry")
    return {
        "uuid": swap_uuid,
        "page_size": 4096,
        "version": version,
        "last_page": last_page,
        "bad_pages": bad_pages,
        "signature": "SWAPSPACE2",
    }
