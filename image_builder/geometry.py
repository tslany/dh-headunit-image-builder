from __future__ import annotations

import hashlib
import os
import struct
from dataclasses import dataclass
from typing import Iterable

from .errors import BuilderError

SECTOR_SIZE = 512
DISK_SECTORS = 234_441_648
DISK_BYTES = DISK_SECTORS * SECTOR_SIZE
EXTENDED_BASE = 8_819_678
EXTENDED_SECTORS = 225_621_970
EXTENDED_END = EXTENDED_BASE + EXTENDED_SECTORS

SYSLINUX_MBR_BYTES = 440
SYSLINUX_MBR_SHA256 = (
    "c5173d2a10111e7ba4debeda7f15c7561800e510352f180ce2ac51aa74fce209"
)


@dataclass(frozen=True)
class Partition:
    number: int
    start: int
    sectors: int
    type_code: int
    role: str
    bootable: bool = False
    ebr_lba: int | None = None

    @property
    def end(self) -> int:
        return self.start + self.sectors

    @property
    def bytes(self) -> int:
        return self.sectors * SECTOR_SIZE


PRIMARY_PARTITIONS = (
    Partition(1, 1, 417_685, 0x83, "boot-active", True),
    Partition(2, 417_686, 6_297_478, 0x83, "root-a"),
    Partition(3, 6_715_164, 2_104_514, 0x82, "swap"),
    Partition(4, EXTENDED_BASE, EXTENDED_SECTORS, 0x05, "extended"),
)

LOGICAL_PARTITIONS = (
    Partition(5, 8_819_679, 417_687, 0x83, "boot-alternate", ebr_lba=8_819_678),
    Partition(6, 9_237_367, 6_297_479, 0x83, "root-b", ebr_lba=9_237_366),
    Partition(7, 15_534_847, 3_148_737, 0x83, "rw-data", ebr_lba=15_534_846),
    Partition(8, 18_683_585, 2_104_513, 0x83, "log-data", ebr_lba=18_683_584),
    Partition(9, 20_788_099, 208_843, 0x83, "backup", ebr_lba=20_788_098),
    Partition(10, 20_996_943, 180_405_425, 0x83, "usr-data", ebr_lba=20_996_942),
    Partition(11, 205_039_734, 8_402_944, 0x83, "voice", ebr_lba=205_037_595),
    Partition(12, 213_444_726, 8_402_944, 0x83, "gracenote", ebr_lba=213_442_678),
    Partition(13, 221_849_718, 10_491_904, 0x83, "common-data", ebr_lba=221_847_670),
    Partition(14, 232_343_670, 2_097_152, 0x83, "reserved-zero", ebr_lba=232_341_622),
)

PARTITIONS = PRIMARY_PARTITIONS + LOGICAL_PARTITIONS
PARTITION_BY_NUMBER = {partition.number: partition for partition in PARTITIONS}

CANONICAL_EBR_SHA256 = {
    8_819_678: "cf648601422167a42eaf373ad1221c437844c3e635cdc4ff19a5afe9247a30e5",
    9_237_366: "fbd2542a62d93ca2909e74a724efa1ecbaba95993de236d75248321f8cdcc9ab",
    15_534_846: "5c7b20f920bfbf7b370d18b090da8aca59091b7a2d384f2d92cbaa35d5831740",
    18_683_584: "de76d9de618369ebea0ad68dd592b1b3f66384fa356f8ec9ab637bea4daf6768",
    20_788_098: "651be2e80dbf745f4fdb94653470bda296fef516888d5694caa83b8b986fa642",
    20_996_942: "669c00b8447d69deb93717c833707abf07916c463c489ca8c5db83739ff8bd1d",
    205_037_595: "4cdb8cddec51edb077793b82a3fb2080f964752103c3fc40022c82bece2ee7ac",
    213_442_678: "5e65326490613bb7fa98d46ad616091f6b12813da54900a1d694329de746f3d8",
    221_847_670: "fa0f1da94430768a27654904fa69de5f8d115e6b602a439ac654e794b97b1c14",
    232_341_622: "174ce13e4d556830c3f40c5c4bf9031b6dfc973645cae8ffe567d8be5e25aaf4",
}


def _chs(lba: int) -> bytes:
    """Encode a legacy 255-head/63-sector CHS address."""
    if lba < 0:
        raise BuilderError(f"negative LBA cannot be encoded: {lba}")
    sectors_per_cylinder = 255 * 63
    cylinder = lba // sectors_per_cylinder
    if cylinder > 1023:
        return b"\xfe\xff\xff"
    remainder = lba % sectors_per_cylinder
    head = remainder // 63
    sector = remainder % 63 + 1
    sector_and_cylinder = sector | ((cylinder >> 2) & 0xC0)
    return bytes((head, sector_and_cylinder, cylinder & 0xFF))


def _entry(
    *,
    bootable: bool,
    type_code: int,
    absolute_start: int,
    relative_start: int,
    sectors: int,
) -> bytes:
    if sectors <= 0:
        raise BuilderError("partition entry must contain at least one sector")
    if relative_start < 0 or relative_start > 0xFFFFFFFF:
        raise BuilderError(f"partition relative start is out of range: {relative_start}")
    if sectors > 0xFFFFFFFF:
        raise BuilderError(f"partition sector count is out of range: {sectors}")
    absolute_end = absolute_start + sectors - 1
    return struct.pack(
        "<B3sB3sII",
        0x80 if bootable else 0,
        _chs(absolute_start),
        type_code,
        _chs(absolute_end),
        relative_start,
        sectors,
    )


def encode_mbr(bootstrap: bytes, disk_id: int) -> bytes:
    if len(bootstrap) != SYSLINUX_MBR_BYTES:
        raise BuilderError(
            f"MBR bootstrap must be {SYSLINUX_MBR_BYTES} bytes, got {len(bootstrap)}"
        )
    if not 0 < disk_id <= 0xFFFFFFFF:
        raise BuilderError("disk ID must be a nonzero unsigned 32-bit value")
    entries = b"".join(
        _entry(
            bootable=partition.bootable,
            type_code=partition.type_code,
            absolute_start=partition.start,
            relative_start=partition.start,
            sectors=partition.sectors,
        )
        for partition in PRIMARY_PARTITIONS
    )
    sector = (
        bootstrap
        + struct.pack("<I", disk_id)
        + b"\0\0"
        + entries
        + b"\x55\xaa"
    )
    if len(sector) != SECTOR_SIZE:
        raise AssertionError(f"internal MBR length error: {len(sector)}")
    return sector


def encode_ebr(index: int) -> bytes:
    partition = LOGICAL_PARTITIONS[index]
    if partition.ebr_lba is None:
        raise AssertionError("logical partition has no EBR LBA")
    relative_data_start = partition.start - partition.ebr_lba
    entries = _entry(
        bootable=False,
        type_code=partition.type_code,
        absolute_start=partition.start,
        relative_start=relative_data_start,
        sectors=partition.sectors,
    )
    if index + 1 < len(LOGICAL_PARTITIONS):
        next_partition = LOGICAL_PARTITIONS[index + 1]
        if next_partition.ebr_lba is None:
            raise AssertionError("next logical partition has no EBR LBA")
        relative_link = next_partition.ebr_lba - EXTENDED_BASE
        link_span = next_partition.start - next_partition.ebr_lba + next_partition.sectors
        entries += _entry(
            bootable=False,
            type_code=0x05,
            absolute_start=next_partition.ebr_lba,
            relative_start=relative_link,
            sectors=link_span,
        )
    else:
        entries += bytes(16)
    sector = bytes(446) + entries + bytes(32) + b"\x55\xaa"
    if len(sector) != SECTOR_SIZE:
        raise AssertionError(f"internal EBR length error: {len(sector)}")
    return sector


def table_sectors(bootstrap: bytes, disk_id: int) -> tuple[tuple[int, bytes], ...]:
    sectors: list[tuple[int, bytes]] = [(0, encode_mbr(bootstrap, disk_id))]
    sectors.extend(
        (partition.ebr_lba, encode_ebr(index))
        for index, partition in enumerate(LOGICAL_PARTITIONS)
        if partition.ebr_lba is not None
    )
    return tuple(sectors)


def validate_definition() -> None:
    if EXTENDED_END != DISK_SECTORS:
        raise AssertionError("extended partition does not end at disk EOF")
    for previous, current in zip(PRIMARY_PARTITIONS, PRIMARY_PARTITIONS[1:]):
        if previous.end > current.start:
            raise AssertionError("overlapping primary partitions")
    previous_ebr = EXTENDED_BASE - 1
    occupied: list[tuple[int, int, str]] = []
    for partition in LOGICAL_PARTITIONS:
        if partition.ebr_lba is None:
            raise AssertionError("logical partition without EBR")
        if not EXTENDED_BASE <= partition.ebr_lba < partition.start:
            raise AssertionError(f"invalid EBR placement for p{partition.number}")
        if partition.ebr_lba <= previous_ebr:
            raise AssertionError("EBR chain is not strictly increasing")
        if partition.end > EXTENDED_END:
            raise AssertionError(f"p{partition.number} exceeds extended partition")
        occupied.append((partition.ebr_lba, partition.ebr_lba + 1, "EBR"))
        occupied.append((partition.start, partition.end, f"p{partition.number}"))
        previous_ebr = partition.ebr_lba
    occupied.sort()
    for left, right in zip(occupied, occupied[1:]):
        if left[1] > right[0]:
            raise AssertionError(f"overlap between {left[2]} and {right[2]}")
    for index, partition in enumerate(LOGICAL_PARTITIONS):
        actual = hashlib.sha256(encode_ebr(index)).hexdigest()
        expected = CANONICAL_EBR_SHA256[partition.ebr_lba]
        if actual != expected:
            raise AssertionError(
                f"canonical EBR p{partition.number} mismatch: {actual} != {expected}"
            )


def pwrite_all(fd: int, data: bytes, offset: int) -> None:
    written = 0
    while written < len(data):
        count = os.pwrite(fd, data[written:], offset + written)
        if count <= 0:
            raise BuilderError("short write while assembling image")
        written += count


def write_table(fd: int, bootstrap: bytes, disk_id: int) -> None:
    for lba, sector in table_sectors(bootstrap, disk_id):
        pwrite_all(fd, sector, lba * SECTOR_SIZE)


def read_exact(fd: int, length: int, offset: int) -> bytes:
    chunks: list[bytes] = []
    done = 0
    while done < length:
        chunk = os.pread(fd, length - done, offset + done)
        if not chunk:
            raise BuilderError(f"unexpected EOF at byte {offset + done}")
        chunks.append(chunk)
        done += len(chunk)
    return b"".join(chunks)


def verify_table(fd: int) -> dict[str, object]:
    mbr = read_exact(fd, SECTOR_SIZE, 0)
    if mbr[510:512] != b"\x55\xaa":
        raise BuilderError("sector zero has no 55 aa MBR signature")
    bootstrap = mbr[:SYSLINUX_MBR_BYTES]
    bootstrap_hash = hashlib.sha256(bootstrap).hexdigest()
    if bootstrap_hash != SYSLINUX_MBR_SHA256:
        raise BuilderError(
            "sector-zero bootstrap is not the supported Syslinux mbr.bin "
            f"({bootstrap_hash})"
        )
    disk_id = struct.unpack_from("<I", mbr, 440)[0]
    if disk_id == 0:
        raise BuilderError("sector-zero disk ID is zero")
    if mbr[444:446] != b"\0\0":
        raise BuilderError("sector-zero reserved bytes 444-445 are nonzero")
    expected_mbr = encode_mbr(bootstrap, disk_id)
    if mbr != expected_mbr:
        raise BuilderError("primary MBR entries differ from the canonical layout")
    ebr_hashes: dict[str, str] = {}
    for index, partition in enumerate(LOGICAL_PARTITIONS):
        assert partition.ebr_lba is not None
        actual = read_exact(fd, SECTOR_SIZE, partition.ebr_lba * SECTOR_SIZE)
        expected = encode_ebr(index)
        if actual != expected:
            raise BuilderError(
                f"p{partition.number} EBR at LBA {partition.ebr_lba} is not canonical"
            )
        ebr_hashes[str(partition.ebr_lba)] = hashlib.sha256(actual).hexdigest()
    return {
        "disk_id": f"0x{disk_id:08x}",
        "mbr_bootstrap_sha256": bootstrap_hash,
        "mbr_sector_sha256": hashlib.sha256(mbr).hexdigest(),
        "ebr_sha256": ebr_hashes,
    }


def unallocated_zero_ranges() -> tuple[tuple[int, int, str], ...]:
    """Return canonical byte ranges which must contain logical zeros."""
    p10 = PARTITION_BY_NUMBER[10]
    p11 = PARTITION_BY_NUMBER[11]
    p12 = PARTITION_BY_NUMBER[12]
    p13 = PARTITION_BY_NUMBER[13]
    p14 = PARTITION_BY_NUMBER[14]
    assert p11.ebr_lba is not None
    assert p12.ebr_lba is not None
    assert p13.ebr_lba is not None
    assert p14.ebr_lba is not None
    sector_ranges = (
        (p10.end, p11.ebr_lba, "p10-to-p11 gap"),
        (p11.ebr_lba + 1, p11.start, "p11 pre-data gap"),
        (p12.ebr_lba + 1, p12.start, "p12 pre-data gap"),
        (p13.ebr_lba + 1, p13.start, "p13 pre-data gap"),
        (p14.ebr_lba + 1, p14.start, "p14 pre-data gap"),
        (p14.start, p14.end, "reserved p14"),
        (p14.end, DISK_SECTORS, "final disk tail"),
    )
    return tuple(
        (start * SECTOR_SIZE, (end - start) * SECTOR_SIZE, description)
        for start, end, description in sector_ranges
        if end > start
    )


def payload_padding_ranges(payload_sizes: dict[int, int]) -> Iterable[tuple[int, int, str]]:
    for number, payload_size in payload_sizes.items():
        if number not in PARTITION_BY_NUMBER:
            raise BuilderError(f"unknown payload partition p{number}")
        partition = PARTITION_BY_NUMBER[number]
        if payload_size < 0 or payload_size > partition.bytes:
            raise BuilderError(
                f"p{number} payload is {payload_size} bytes but capacity is "
                f"{partition.bytes}"
            )
        padding = partition.bytes - payload_size
        if padding:
            yield (
                partition.start * SECTOR_SIZE + payload_size,
                padding,
                f"p{number} payload padding",
            )


validate_definition()
