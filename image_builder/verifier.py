from __future__ import annotations

import os
import re
import struct
import uuid as uuid_module
from typing import Callable, Mapping

from .contracts import INPUT_ROLES, RESTORED_FILESYSTEM_SPECS
from .errors import BuilderError
from .extfs import (
    EXT_STATE_CLEAN,
    fresh_ext_spec,
    parse_superblock,
    validate_superblock,
)
from .geometry import (
    DISK_BYTES,
    PARTITION_BY_NUMBER,
    SECTOR_SIZE,
    verify_table,
)
from .io_utils import sha256_range_fd

IMMUTABLE_PARTITIONS = {
    1: "boot",
    2: "root",
    5: "boot",
    6: "root",
    11: "vr",
}
ROLE_PRIMARY_PARTITION = {"boot": "1", "root": "2", "vr": "11"}
FRESH_EXT_PARTITIONS = (7, 8, 9, 10, 12, 13)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
Progress = Callable[[str], None]


def _verify_assembled_image(
    fd: int,
    size: int,
    *,
    expected_payload_sha256: Mapping[str, str],
    progress: Progress | None = None,
) -> dict[str, object]:
    if size != DISK_BYTES:
        raise BuilderError(
            f"image size is {size}, expected exactly {DISK_BYTES} bytes"
        )
    table = verify_table(fd)

    filesystems: dict[str, object] = {}
    for number, role in IMMUTABLE_PARTITIONS.items():
        partition = PARTITION_BY_NUMBER[number]
        spec = RESTORED_FILESYSTEM_SPECS[role]
        superblock = parse_superblock(fd, partition.start * SECTOR_SIZE)
        validate_superblock(
            superblock,
            spec,
            description=f"p{number} {role}",
            container_bytes=partition.bytes,
        )
        if superblock.filesystem_bytes != spec["bytes"]:
            raise BuilderError(
                f"p{number} {role} filesystem byte size is "
                f"{superblock.filesystem_bytes}, expected {spec['bytes']}"
            )
        filesystems[str(number)] = {
            "role": role,
            **superblock.as_dict(),
        }

    partition_hashes: dict[str, str] = {}
    for number, role in IMMUTABLE_PARTITIONS.items():
        if progress:
            progress(f"hashing embedded p{number} {role} filesystem")
        partition = PARTITION_BY_NUMBER[number]
        partition_hashes[str(number)] = sha256_range_fd(
            fd,
            partition.start * SECTOR_SIZE,
            int(RESTORED_FILESYSTEM_SPECS[role]["bytes"]),
        )

    if partition_hashes["1"] != partition_hashes["5"]:
        raise BuilderError("p1 and p5 boot filesystem copies differ")
    if partition_hashes["2"] != partition_hashes["6"]:
        raise BuilderError("p2 and p6 root filesystem copies differ")

    payload_hashes = {
        role: partition_hashes[partition]
        for role, partition in ROLE_PRIMARY_PARTITION.items()
    }
    expected = _validate_expected_payload_hashes(
        expected_payload_sha256
    )
    for role in INPUT_ROLES:
        if payload_hashes[role] != expected[role]:
            raise BuilderError(
                f"assembled {role} filesystem differs from the "
                "verified build payload"
            )

    for number in FRESH_EXT_PARTITIONS:
        partition = PARTITION_BY_NUMBER[number]
        expected = fresh_ext_spec(number)
        superblock = parse_superblock(fd, partition.start * SECTOR_SIZE)
        _require_superblock_values(
            number,
            superblock,
            inode_count=expected["inode_count"],
            block_count=expected["block_count"],
            reserved_block_count=expected["block_count"] * 5 // 100,
            block_size=expected["block_size"],
            blocks_per_group=expected["blocks_per_group"],
            inodes_per_group=expected["inodes_per_group"],
            inode_size=expected["inode_size"],
            feature_compat=expected["feature_compat"],
            feature_incompat=expected["feature_incompat"],
            feature_ro_compat=expected["feature_ro_compat"],
            journal_bytes=int(expected["journal_size_mb"]) * 1024 * 1024,
            default_mount_options=expected["default_mount_options"],
            min_extra_isize=expected["min_extra_isize"],
            want_extra_isize=expected["want_extra_isize"],
        )
        expected_bytes = int(expected["block_count"]) * int(
            expected["block_size"]
        )
        if superblock.filesystem_bytes != expected_bytes:
            raise BuilderError(
                f"p{number} filesystem is "
                f"{superblock.filesystem_bytes} bytes, "
                f"expected exactly {expected_bytes}"
            )
        if superblock.filesystem_bytes > partition.bytes:
            raise BuilderError(f"p{number} filesystem exceeds its partition")
        filesystems[str(number)] = {
            "role": partition.role,
            **superblock.as_dict(),
        }

    return {
        "schema": "genesis-image-partition-verification/v1",
        "valid": True,
        "image_bytes": size,
        "table": table,
        "partition_payload_sha256": partition_hashes,
        "payload_sha256": payload_hashes,
        "filesystems": filesystems,
        "swap": _parse_swap(fd),
    }


def _validate_expected_payload_hashes(
    value: Mapping[str, str],
) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise BuilderError("expected payload hashes must be a mapping")
    if set(value) != set(INPUT_ROLES):
        raise BuilderError(
            "expected payload hashes must contain exactly boot, root, and vr"
        )
    for role in INPUT_ROLES:
        digest = value[role]
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            raise BuilderError(
                f"expected {role} payload SHA-256 is invalid"
            )
    return value


def _parse_swap(fd: int) -> dict[str, object]:
    partition = PARTITION_BY_NUMBER[3]
    base = partition.start * SECTOR_SIZE
    header = os.pread(fd, 4096, base)
    if len(header) != 4096:
        raise BuilderError("p3 swap header is truncated")
    if header[-10:] != b"SWAPSPACE2":
        raise BuilderError(
            "p3 does not contain a 4096-byte-page SWAPSPACE2 header"
        )
    version, last_page, bad_pages = struct.unpack_from("<III", header, 1024)
    if version != 1:
        raise BuilderError(f"p3 swap version is {version}, expected 1")
    expected_last_page = partition.bytes // 4096 - 1
    if last_page != expected_last_page:
        raise BuilderError(
            f"p3 swap last page is {last_page}, "
            f"expected {expected_last_page}"
        )
    if bad_pages != 0:
        raise BuilderError(f"p3 swap declares {bad_pages} bad pages")
    swap_uuid = str(uuid_module.UUID(bytes=bytes(header[1036:1052])))
    if swap_uuid == str(uuid_module.UUID(int=0)):
        raise BuilderError("p3 swap UUID is zero")
    return {
        "partition": 3,
        "signature": "SWAPSPACE2",
        "page_size": 4096,
        "version": version,
        "last_page": last_page,
        "bad_pages": bad_pages,
        "uuid": swap_uuid,
    }


def _require_superblock_values(
    number: int,
    superblock: object,
    **expected: object,
) -> None:
    for key, value in expected.items():
        actual = getattr(superblock, key)
        if actual != value:
            raise BuilderError(
                f"p{number} ext superblock has {key}={actual!r}, "
                f"expected {value!r}"
            )
    if not getattr(superblock, "state") & EXT_STATE_CLEAN:
        raise BuilderError(f"p{number} ext filesystem is not marked clean")
