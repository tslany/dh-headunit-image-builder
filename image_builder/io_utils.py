from __future__ import annotations

import errno
import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .errors import BuilderError

COPY_CHUNK = 8 * 1024 * 1024
HASH_CHUNK = 8 * 1024 * 1024


@dataclass
class OpenedRegular:
    path: Path
    fd: int
    size: int
    identity: tuple[int, int, int, int, int]

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def __enter__(self) -> "OpenedRegular":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def assert_unchanged(self) -> None:
        current = os.fstat(self.fd)
        identity = (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        )
        if identity != self.identity:
            raise BuilderError(f"input changed while in use: {self.path}")


def open_regular_read(path_value: str | os.PathLike[str]) -> OpenedRegular:
    path = Path(path_value)
    if str(path) == "-":
        raise BuilderError("standard input is not accepted")
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise BuilderError(f"cannot inspect input {path}: {exc.strerror}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise BuilderError(f"input is not a regular file: {path}")
    flags = os.O_RDONLY | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise BuilderError(f"cannot open input {path}: {exc.strerror}") from exc
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise BuilderError(f"opened input is not a regular file: {path}")
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise BuilderError(f"input was replaced while opening: {path}")
        return OpenedRegular(
            path=path,
            fd=fd,
            size=opened.st_size,
            identity=(
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
            ),
        )
    except Exception:
        os.close(fd)
        raise


def sha256_fd(fd: int, size: int, progress: Callable[[int, int], None] | None = None) -> str:
    return sha256_range_fd(fd, 0, size, progress=progress)


def sha256_range_fd(
    fd: int,
    offset: int,
    size: int,
    progress: Callable[[int, int], None] | None = None,
) -> str:
    _require_bounded_range(fd, offset, size, "hash")
    digest = hashlib.sha256()
    position = 0
    while position < size:
        chunk = os.pread(
            fd,
            min(HASH_CHUNK, size - position),
            offset + position,
        )
        if not chunk:
            raise BuilderError(
                f"unexpected EOF while hashing at byte {offset + position}"
            )
        digest.update(chunk)
        position += len(chunk)
        if progress is not None:
            progress(position, size)
    return digest.hexdigest()


def pwrite_all(fd: int, data: bytes, offset: int) -> None:
    done = 0
    while done < len(data):
        written = os.pwrite(fd, data[done:], offset + done)
        if written <= 0:
            raise BuilderError(f"short write at output byte {offset + done}")
        done += written


def _data_extents(fd: int, size: int) -> Iterable[tuple[int, int]]:
    seek_data = getattr(os, "SEEK_DATA", None)
    seek_hole = getattr(os, "SEEK_HOLE", None)
    if seek_data is None or seek_hole is None:
        raise OSError(errno.EOPNOTSUPP, "SEEK_DATA/SEEK_HOLE unavailable")
    position = 0
    while position < size:
        try:
            data = os.lseek(fd, position, seek_data)
        except OSError as exc:
            if exc.errno == errno.ENXIO:
                break
            raise
        if data >= size:
            break
        hole = os.lseek(fd, data, seek_hole)
        if hole <= data:
            raise BuilderError("filesystem returned an invalid sparse extent")
        yield data, min(hole, size)
        position = hole


def copy_sparse_to_offsets(
    source_fd: int,
    destination_fd: int,
    source_size: int,
    destination_offsets: Iterable[int],
) -> dict[str, int | str]:
    offsets = tuple(destination_offsets)
    if not offsets:
        raise BuilderError("sparse copy has no destination offsets")
    _require_bounded_range(source_fd, 0, source_size, "sparse-copy source")
    destination_size = os.fstat(destination_fd).st_size
    for destination_offset in offsets:
        if destination_offset < 0:
            raise BuilderError("sparse-copy destination offset is negative")
        if destination_offset + source_size > destination_size:
            raise BuilderError(
                "sparse copy exceeds destination: "
                f"{destination_offset} + {source_size} > {destination_size}"
            )
    data_bytes = 0
    extent_count = 0
    fallback = False
    try:
        extents = _data_extents(source_fd, source_size)
        for start, end in extents:
            extent_count += 1
            data_bytes += _copy_extent(
                source_fd,
                destination_fd,
                start,
                end,
                offsets,
                skip_zero_chunks=False,
            )
    except OSError as exc:
        if exc.errno not in (errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP):
            raise
        if extent_count:
            raise BuilderError(
                "sparse extent discovery failed after copying began"
            ) from exc
        fallback = True
        extent_count = 1
        data_bytes = _copy_extent(
            source_fd,
            destination_fd,
            0,
            source_size,
            offsets,
            skip_zero_chunks=True,
        )
    return {
        "method": "zero-detecting-fallback" if fallback else "seek-data-hole",
        "source_data_bytes": data_bytes,
        "source_extents": extent_count,
    }


def copy_sparse_range(
    source_fd: int,
    destination_fd: int,
    source_offset: int,
    length: int,
    destination_offset: int = 0,
) -> dict[str, int | str]:
    """Copy one bounded source range into an all-zero destination range."""
    _require_bounded_range(source_fd, source_offset, length, "sparse-range source")
    _require_bounded_range(
        destination_fd,
        destination_offset,
        length,
        "sparse-range destination",
    )
    if not range_is_zero(destination_fd, destination_offset, length):
        raise BuilderError("sparse-range destination must be all zero")
    data_bytes = 0
    extent_count = 0
    fallback = False
    destination_shift = destination_offset - source_offset
    end = source_offset + length
    try:
        extents = _data_extents_in_range(source_fd, source_offset, end)
        for start, extent_end in extents:
            extent_count += 1
            data_bytes += _copy_extent(
                source_fd,
                destination_fd,
                start,
                extent_end,
                (destination_shift,),
                skip_zero_chunks=True,
            )
    except OSError as exc:
        if exc.errno not in (errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP):
            raise
        if extent_count:
            raise BuilderError(
                "sparse range discovery failed after copying began"
            ) from exc
        fallback = True
        extent_count = 1 if length else 0
        data_bytes = _copy_extent(
            source_fd,
            destination_fd,
            source_offset,
            end,
            (destination_shift,),
            skip_zero_chunks=True,
        )
    return {
        "method": "zero-detecting-fallback" if fallback else "seek-data-hole",
        "source_data_bytes": data_bytes,
        "source_extents": extent_count,
    }


def _copy_extent(
    source_fd: int,
    destination_fd: int,
    start: int,
    end: int,
    destination_offsets: tuple[int, ...],
    *,
    skip_zero_chunks: bool,
) -> int:
    data_bytes = 0
    position = start
    while position < end:
        chunk = os.pread(source_fd, min(COPY_CHUNK, end - position), position)
        if not chunk:
            raise BuilderError(f"unexpected EOF while copying at byte {position}")
        if skip_zero_chunks and not any(chunk):
            position += len(chunk)
            continue
        for destination_offset in destination_offsets:
            pwrite_all(destination_fd, chunk, destination_offset + position)
        data_bytes += len(chunk)
        position += len(chunk)
    return data_bytes


def range_is_zero(fd: int, offset: int, length: int) -> bool:
    _require_bounded_range(fd, offset, length, "zero check")
    end = offset + length
    if length == 0:
        return True
    try:
        extents = _data_extents_in_range(fd, offset, end)
        for start, extent_end in extents:
            if not _bytes_are_zero(fd, start, extent_end - start):
                return False
        return True
    except OSError as exc:
        if exc.errno not in (errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP):
            raise
        return _bytes_are_zero(fd, offset, length)


def _data_extents_in_range(fd: int, start: int, end: int) -> Iterable[tuple[int, int]]:
    seek_data = getattr(os, "SEEK_DATA", None)
    seek_hole = getattr(os, "SEEK_HOLE", None)
    if seek_data is None or seek_hole is None:
        raise OSError(errno.EOPNOTSUPP, "SEEK_DATA/SEEK_HOLE unavailable")
    position = start
    while position < end:
        try:
            data = os.lseek(fd, position, seek_data)
        except OSError as exc:
            if exc.errno == errno.ENXIO:
                break
            raise
        if data >= end:
            break
        hole = os.lseek(fd, data, seek_hole)
        if hole <= data:
            raise BuilderError("filesystem returned an invalid sparse extent")
        yield data, min(hole, end)
        position = hole


def _bytes_are_zero(fd: int, offset: int, length: int) -> bool:
    position = 0
    while position < length:
        chunk = os.pread(fd, min(COPY_CHUNK, length - position), offset + position)
        if not chunk:
            raise BuilderError(f"unexpected EOF while checking zero range at {offset}")
        if any(chunk):
            return False
        position += len(chunk)
    return True


def _require_bounded_range(fd: int, offset: int, length: int, operation: str) -> None:
    if offset < 0 or length < 0:
        raise BuilderError(f"{operation} range cannot be negative")
    file_size = os.fstat(fd).st_size
    if offset + length > file_size:
        raise BuilderError(
            f"{operation} range exceeds file: {offset} + {length} > {file_size}"
        )


def validate_new_output_path(path_value: str | os.PathLike[str]) -> Path:
    path = Path(path_value)
    if str(path) == "-":
        raise BuilderError("standard output is not accepted as an image destination")
    if not path.name:
        raise BuilderError("output path has no filename")
    try:
        existing = os.lstat(path)
    except FileNotFoundError:
        existing = None
    except OSError as exc:
        raise BuilderError(f"cannot inspect output path {path}: {exc.strerror}") from exc
    if existing is not None:
        kind = "block device" if stat.S_ISBLK(existing.st_mode) else "existing path"
        raise BuilderError(f"refusing {kind} as output: {path}")
    parent = path.parent if str(path.parent) else Path(".")
    try:
        parent_stat = os.stat(parent)
    except OSError as exc:
        raise BuilderError(f"cannot inspect output directory {parent}: {exc.strerror}") from exc
    if not stat.S_ISDIR(parent_stat.st_mode):
        raise BuilderError(f"output parent is not a directory: {parent}")
    return path


def publish_no_replace(temporary_path: Path, final_path: Path) -> None:
    try:
        directory_fd = os.open(
            final_path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
        )
    except OSError as exc:
        raise BuilderError(
            f"cannot open output directory {final_path.parent}: {exc.strerror}"
        ) from exc
    published = False
    try:
        try:
            os.link(temporary_path, final_path, follow_symlinks=False)
            published = True
        except FileExistsError as exc:
            raise BuilderError(
                f"output appeared during build; refusing overwrite: {final_path}"
            ) from exc
        except OSError as exc:
            raise BuilderError(
                f"cannot publish output {final_path}: {exc.strerror}"
            ) from exc
        os.fsync(directory_fd)
        os.unlink(temporary_path)
        os.fsync(directory_fd)
    except BaseException as exc:
        rollback_error: OSError | None = None
        if published:
            try:
                os.unlink(final_path)
                os.fsync(directory_fd)
            except OSError as cleanup_exc:
                rollback_error = cleanup_exc
        if rollback_error is not None:
            raise BuilderError(
                f"publication failed and rollback of {final_path} also failed: "
                f"{rollback_error.strerror}"
            ) from exc
        raise
    finally:
        os.close(directory_fd)
