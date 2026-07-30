from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Sequence

from .errors import BuilderError
from .extfs import (
    debugfs_cat,
    debugfs_replace_regular,
    debugfs_require_path,
    run_e2fsck_path,
)
from .io_utils import OpenedRegular, open_regular_read


PATCH_SCHEMA = "dh-headunit-application-patch/v2"
PATCH_REPORT_SCHEMA = "dh-headunit-byte-patch-report/v1"
MAX_PATCH_BYTES = 1024 * 1024
MAX_TARGET_BYTES = 64 * 1024 * 1024
MAX_TARGETS_PER_PATCH = 4
MAX_HUNKS_PER_TARGET = 64
MAX_PATCH_FILES = 128
MAX_SELECTED_TARGETS = 128
MAX_SELECTED_HUNKS = 1024
MAX_SELECTED_PATCH_BYTES = 8 * 1024 * 1024
MAX_CHANGED_BYTES = 64 * 1024

# These are directories stored in the restored p2 root filesystem.  Other
# top-level paths used by the running head unit are separate filesystems or
# mount points and must not be addressed by an offline root-image patch.
ROOT_TARGET_PREFIXES = (
    "/app/",
    "/bin/",
    "/etc/",
    "/lib/",
    "/sbin/",
    "/usr/",
)

_ID_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_HEX_RE = re.compile(r"(?:[0-9a-f]{2})+\Z")
_PATH_COMPONENT_RE = re.compile(r"[A-Za-z0-9._+-]+\Z")
Progress = Callable[[str], None]


@dataclass(frozen=True)
class PatchHunk:
    patch_id: str
    index: int
    offset: int
    before: bytes
    after: bytes


@dataclass(frozen=True)
class PatchTarget:
    path: str
    source_bytes: int
    source_sha256: str
    hunks: tuple[PatchHunk, ...]


@dataclass(frozen=True)
class PatchRecipe:
    patch_id: str
    revision: int | None
    file_bytes: int
    definition_sha256: str
    targets: tuple[PatchTarget, ...]
    changed_bytes: int


@dataclass(frozen=True)
class _TargetPlan:
    path: str
    source_bytes: int
    source_sha256: str
    hunks: tuple[PatchHunk, ...]


@dataclass(frozen=True)
class _HunkState:
    patch_id: str
    index: int
    offset: int
    size: int
    state: str


@dataclass(frozen=True)
class _TargetSnapshot:
    path: str
    source_bytes: int
    source_sha256: str
    current_sha256: str
    desired_sha256: str
    reconstructed_sha256: str
    mode: int
    uid: int
    gid: int
    hunk_states: tuple[_HunkState, ...]


def load_patch_catalog(
    patch_directory: str | os.PathLike[str],
    excluded_patch_ids: Iterable[str] = (),
) -> tuple[PatchRecipe, ...]:
    """Load a sorted external SD-patcher catalog, minus explicit exclusions."""

    directory = Path(patch_directory)
    try:
        directory_stat = os.lstat(directory)
    except OSError as exc:
        raise BuilderError(
            f"cannot inspect patch directory {directory}: {exc.strerror}"
        ) from exc
    if stat.S_ISLNK(directory_stat.st_mode) or not stat.S_ISDIR(
        directory_stat.st_mode
    ):
        raise BuilderError(f"patch directory is missing or unsafe: {directory}")
    try:
        names = sorted(
            name for name in os.listdir(directory) if name.endswith(".json")
        )
    except OSError as exc:
        raise BuilderError(
            f"cannot list patch directory {directory}: {exc.strerror}"
        ) from exc
    if not names:
        _normalize_exclusions(excluded_patch_ids, ())
        return ()
    if len(names) > MAX_PATCH_FILES:
        raise BuilderError("patch directory contains more than 128 patches")

    available_ids: list[str] = []
    for name in names:
        patch_id = name[: -len(".json")]
        if _ID_RE.fullmatch(patch_id) is None:
            raise BuilderError(f"invalid patch filename: {name}")
        available_ids.append(patch_id)

    excluded = _normalize_exclusions(excluded_patch_ids, available_ids)
    recipes: list[PatchRecipe] = []
    selected_definition_bytes = 0
    selected_changed_bytes = 0
    selected_target_count = 0
    selected_hunk_count = 0
    for patch_id in available_ids:
        recipe = _read_patch(
            directory / f"{patch_id}.json",
            patch_id,
            revision_required=True,
        )
        if patch_id in excluded:
            continue
        recipes.append(recipe)
        selected_definition_bytes += recipe.file_bytes
        selected_changed_bytes += recipe.changed_bytes
        selected_target_count += len(recipe.targets)
        selected_hunk_count += sum(len(target.hunks) for target in recipe.targets)
        if selected_definition_bytes > MAX_SELECTED_PATCH_BYTES:
            raise BuilderError("selected patch definitions exceed 8 MiB")
        if selected_changed_bytes > MAX_CHANGED_BYTES:
            raise BuilderError("selected patches change more than 64 KiB")
        if (
            selected_target_count > MAX_SELECTED_TARGETS
            or selected_hunk_count > MAX_SELECTED_HUNKS
        ):
            raise BuilderError("selected patch set exceeds aggregate limits")

    # Build the combined plans now so source disagreements and cross-recipe
    # overlaps fail before the caller starts constructing an image.
    _build_target_plans(recipes)
    return tuple(recipes)


def patch_target_paths(patches: Sequence[PatchRecipe]) -> set[str]:
    """Return the root-filesystem paths addressed by a validated patch set."""

    return set(_build_target_plans(patches))


def apply_external_patches(
    root_path: Path,
    patches: Sequence[PatchRecipe],
    *,
    progress: Progress | None = None,
) -> list[dict[str, object]]:
    """Apply selected recipes to an already-private p2 ext image.

    Every hunk may initially contain either its stock or replacement bytes.
    Mixed states are accepted only when replacing all recognized hunks with
    their stock bytes reconstructs the recipe's full source fingerprint.
    """

    plans = _build_target_plans(patches)
    if not plans:
        return []
    root_path = Path(root_path)

    snapshots: dict[str, _TargetSnapshot] = {}
    for path in sorted(plans):
        if progress is not None:
            progress(f"checking external byte patches for {path}")
        snapshots[path] = _inspect_target(root_path, plans[path])

    for path in sorted(plans):
        plan = plans[path]
        snapshot = snapshots[path]
        data, metadata = _read_target(root_path, plan)
        current_sha256 = _sha256(data)
        if (
            current_sha256 != snapshot.current_sha256
            or metadata["mode"] != snapshot.mode
            or metadata["uid"] != snapshot.uid
            or metadata["gid"] != snapshot.gid
        ):
            raise BuilderError(f"patch target changed after preflight: {path}")
        desired, reconstructed_sha256, hunk_states = _transform_target(data, plan)
        if (
            reconstructed_sha256 != snapshot.reconstructed_sha256
            or _sha256(desired) != snapshot.desired_sha256
            or hunk_states != snapshot.hunk_states
        ):
            raise BuilderError(f"patch target state changed after preflight: {path}")

        if current_sha256 != snapshot.desired_sha256:
            if progress is not None:
                progress(f"applying external byte patches to {path}")
            _replace_target(
                root_path,
                path,
                desired,
                mode=snapshot.mode,
                uid=snapshot.uid,
                gid=snapshot.gid,
            )
        del desired
        del data
        _verify_target_after(root_path, plan, snapshot)

    if progress is not None:
        progress("checking root filesystem after external byte patches")
    run_e2fsck_path(root_path, "external-byte-patched restored root")

    final_snapshots: dict[str, _TargetSnapshot] = {}
    for path in sorted(plans):
        final = _inspect_target(root_path, plans[path])
        expected = snapshots[path]
        if (
            final.current_sha256 != expected.desired_sha256
            or final.desired_sha256 != expected.desired_sha256
            or any(state.state != "replacement" for state in final.hunk_states)
            or final.mode != expected.mode
            or final.uid != expected.uid
            or final.gid != expected.gid
        ):
            raise BuilderError(f"final patch verification failed for {path}")
        final_snapshots[path] = final

    return _build_reports(patches, snapshots, final_snapshots)


def _normalize_exclusions(
    values: Iterable[str], available_ids: Sequence[str]
) -> set[str]:
    if isinstance(values, (str, bytes)):
        raise BuilderError("excluded patch IDs must be an iterable of IDs")
    available = set(available_ids)
    excluded: set[str] = set()
    try:
        iterator = iter(values)
    except TypeError as exc:
        raise BuilderError("excluded patch IDs must be iterable") from exc
    for value in iterator:
        if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
            raise BuilderError("excluded patch IDs must be valid nonempty IDs")
        if value in excluded:
            raise BuilderError(f"duplicate excluded patch: {value}")
        if value not in available:
            choices = ", ".join(available_ids)
            raise BuilderError(
                f"unknown excluded patch {value!r}; available patches: {choices}"
            )
        excluded.add(value)
    return excluded


def _read_patch(
    path: Path,
    patch_id: str,
    *,
    revision_required: bool = False,
) -> PatchRecipe:
    with open_regular_read(path) as opened:
        if not 1 <= opened.size <= MAX_PATCH_BYTES:
            raise BuilderError(f"patch file size is invalid: {path}")
        raw = os.pread(opened.fd, opened.size, 0)
        opened.assert_unchanged()
    if len(raw) != opened.size:
        raise BuilderError(f"patch changed while being read: {path}")
    if b"\0" in raw:
        raise BuilderError(f"patch contains a NUL byte: {path}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BuilderError(f"patch is not valid UTF-8: {path}") from exc

    try:
        value = json.loads(text, object_pairs_hook=_reject_duplicate_json_keys)
    except (TypeError, ValueError) as exc:
        raise BuilderError(f"patch is invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BuilderError(f"{patch_id} must be a JSON object")
    document = {key: item for key, item in value.items() if key != "_comment"}
    if "revision" in document:
        _json_object(document, patch_id, ("schema", "revision", "targets"))
        revision = _json_integer(
            document["revision"], f"{patch_id} revision", 1, 0x7FFFFFFF
        )
    elif revision_required:
        raise BuilderError(f"{patch_id} is missing field revision")
    else:
        _json_object(document, patch_id, ("schema", "targets"))
        revision = None
    if document["schema"] != PATCH_SCHEMA:
        raise BuilderError(f"{patch_id} uses an unsupported patch schema")
    targets_value = document["targets"]
    if not isinstance(targets_value, list) or not targets_value:
        raise BuilderError(f"{patch_id} targets must be a nonempty JSON array")
    if len(targets_value) > MAX_TARGETS_PER_PATCH:
        raise BuilderError(f"{patch_id} has too many targets")

    targets: list[PatchTarget] = []
    previous_path = ""
    changed_bytes = 0
    for target_index, target_value in enumerate(targets_value):
        location = f"{patch_id} target {target_index}"
        _json_object(
            target_value,
            location,
            ("path", "source", "hunks"),
        )
        target_path = _json_text(target_value["path"], f"{location} path", 512)
        if not _safe_target_path(target_path):
            raise BuilderError(f"{location} has an unsafe p2 target path")
        if target_path <= previous_path:
            raise BuilderError(f"{patch_id} target paths are not sorted and unique")
        source_value = target_value["source"]
        _json_object(source_value, f"{location} source", ("bytes", "sha256"))
        source_bytes = _json_integer(
            source_value["bytes"], f"{location} source bytes", 1, MAX_TARGET_BYTES
        )
        source_sha256 = _json_text(
            source_value["sha256"], f"{location} source sha256", 64
        )
        if _SHA256_RE.fullmatch(source_sha256) is None:
            raise BuilderError(f"{location} has an invalid source SHA-256")
        hunks_value = target_value["hunks"]
        if not isinstance(hunks_value, list) or not hunks_value:
            raise BuilderError(f"{location} hunks must be a nonempty JSON array")
        if len(hunks_value) > MAX_HUNKS_PER_TARGET:
            raise BuilderError(f"{patch_id} target has too many hunks")
        current_hunks: list[PatchHunk] = []
        previous_end = 0
        for hunk_index, hunk_value in enumerate(hunks_value):
            hunk_location = f"{location} hunk {hunk_index}"
            _json_object(
                hunk_value,
                hunk_location,
                ("file_offset", "expected_hex", "replacement_hex"),
            )
            offset = _json_integer(
                hunk_value["file_offset"],
                f"{hunk_location} file_offset",
                0,
                source_bytes,
            )
            before_value = _json_text(
                hunk_value["expected_hex"],
                f"{hunk_location} expected_hex",
                MAX_PATCH_BYTES * 2,
            )
            after_value = _json_text(
                hunk_value["replacement_hex"],
                f"{hunk_location} replacement_hex",
                MAX_PATCH_BYTES * 2,
            )
            before, after = _hunk_bytes(before_value, after_value, hunk_location)
            end = offset + len(before)
            if offset < previous_end:
                raise BuilderError(f"{patch_id} hunks overlap or are unsorted")
            if end > source_bytes:
                raise BuilderError(f"{hunk_location} exceeds its target")
            current_hunks.append(
                PatchHunk(
                    patch_id=patch_id,
                    index=len(current_hunks),
                    offset=offset,
                    before=before,
                    after=after,
                )
            )
            previous_end = end
            changed_bytes += len(before)
            if changed_bytes > MAX_CHANGED_BYTES:
                raise BuilderError(f"{patch_id} changes more than 64 KiB")
        targets.append(
            PatchTarget(
                path=target_path,
                source_bytes=source_bytes,
                source_sha256=source_sha256,
                hunks=tuple(current_hunks),
            )
        )
        previous_path = target_path
    return PatchRecipe(
        patch_id=patch_id,
        revision=revision,
        file_bytes=len(raw),
        definition_sha256=_sha256(raw),
        targets=tuple(targets),
        changed_bytes=changed_bytes,
    )


def _json_object(value: object, location: str, required: tuple[str, ...]) -> None:
    if not isinstance(value, dict):
        raise BuilderError(f"{location} must be a JSON object")
    keys = set(value)
    expected = set(required)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        if missing:
            raise BuilderError(f"{location} is missing field {missing[0]}")
        raise BuilderError(f"{location} has unknown field {extra[0]}")


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise BuilderError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _json_text(value: object, description: str, maximum: int) -> str:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > maximum:
        raise BuilderError(f"{description} must be nonempty UTF-8 text")
    if "\0" in value:
        raise BuilderError(f"{description} contains a NUL byte")
    return value


def _json_integer(value: object, description: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BuilderError(f"{description} must be a JSON integer")
    if not minimum <= value <= maximum:
        raise BuilderError(f"{description} is outside the allowed range")
    return value


def _hunk_bytes(
    before_value: str, after_value: str, description: str
) -> tuple[bytes, bytes]:
    if (
        _HEX_RE.fullmatch(before_value) is None
        or _HEX_RE.fullmatch(after_value) is None
    ):
        raise BuilderError(
            f"{description} must use nonempty lowercase hexadecimal"
        )
    before = bytes.fromhex(before_value)
    after = bytes.fromhex(after_value)
    if len(before) != len(after):
        raise BuilderError(f"{description} must be fixed-length")
    if before == after:
        raise BuilderError(f"{description} changes no bytes")
    return before, after


def _safe_target_path(value: str) -> bool:
    if not value.startswith(ROOT_TARGET_PREFIXES):
        return False
    path = PurePosixPath(value)
    return (
        str(path) == value
        and all(
            component not in ("", ".", "..")
            and _PATH_COMPONENT_RE.fullmatch(component) is not None
            for component in path.parts[1:]
        )
    )


def _build_target_plans(
    patches: Sequence[PatchRecipe],
) -> dict[str, _TargetPlan]:
    mutable: dict[str, dict[str, object]] = {}
    previous_id = ""
    target_count = 0
    hunk_count = 0
    definition_bytes = 0
    changed_bytes = 0
    for recipe in patches:
        if not isinstance(recipe, PatchRecipe):
            raise BuilderError("patch set contains an invalid recipe")
        if recipe.patch_id <= previous_id:
            raise BuilderError("patch recipes are not sorted and unique")
        previous_id = recipe.patch_id
        definition_bytes += recipe.file_bytes
        changed_bytes += recipe.changed_bytes
        target_count += len(recipe.targets)
        hunk_count += sum(len(target.hunks) for target in recipe.targets)
        for target in recipe.targets:
            current = mutable.get(target.path)
            if current is None:
                current = {
                    "source_bytes": target.source_bytes,
                    "source_sha256": target.source_sha256,
                    "hunks": [],
                }
                mutable[target.path] = current
            elif (
                current["source_bytes"] != target.source_bytes
                or current["source_sha256"] != target.source_sha256
            ):
                raise BuilderError(
                    f"selected patches disagree about source {target.path}"
                )
            hunks = current["hunks"]
            assert isinstance(hunks, list)
            hunks.extend(target.hunks)

    if definition_bytes > MAX_SELECTED_PATCH_BYTES:
        raise BuilderError("selected patch definitions exceed 8 MiB")
    if changed_bytes > MAX_CHANGED_BYTES:
        raise BuilderError("selected patches change more than 64 KiB")
    if target_count > MAX_SELECTED_TARGETS or hunk_count > MAX_SELECTED_HUNKS:
        raise BuilderError("selected patch set exceeds aggregate limits")

    plans: dict[str, _TargetPlan] = {}
    for path, values in mutable.items():
        raw_hunks = values["hunks"]
        assert isinstance(raw_hunks, list)
        hunks = tuple(sorted(raw_hunks, key=lambda item: item.offset))
        previous_end = 0
        previous_patch: str | None = None
        for hunk in hunks:
            if hunk.offset < previous_end:
                raise BuilderError(
                    f"patches {previous_patch} and {hunk.patch_id} overlap in {path}"
                )
            previous_end = hunk.offset + len(hunk.before)
            previous_patch = hunk.patch_id
        source_bytes = values["source_bytes"]
        source_sha256 = values["source_sha256"]
        assert isinstance(source_bytes, int)
        assert isinstance(source_sha256, str)
        plans[path] = _TargetPlan(
            path=path,
            source_bytes=source_bytes,
            source_sha256=source_sha256,
            hunks=hunks,
        )
    return plans


def _inspect_target(root_path: Path, plan: _TargetPlan) -> _TargetSnapshot:
    data, metadata = _read_target(root_path, plan)
    desired, reconstructed_sha256, states = _transform_target(data, plan)
    snapshot = _TargetSnapshot(
        path=plan.path,
        source_bytes=plan.source_bytes,
        source_sha256=plan.source_sha256,
        current_sha256=_sha256(data),
        desired_sha256=_sha256(desired),
        reconstructed_sha256=reconstructed_sha256,
        mode=_metadata_integer(metadata, "mode", plan.path, 0, 0o777),
        uid=_metadata_integer(metadata, "uid", plan.path, 0, 0xFFFFFFFF),
        gid=_metadata_integer(metadata, "gid", plan.path, 0, 0xFFFFFFFF),
        hunk_states=states,
    )
    del desired
    del data
    return snapshot


def _read_target(
    root_path: Path, plan: _TargetPlan
) -> tuple[bytes, dict[str, object]]:
    with open_regular_read(root_path) as root:
        _require_target_ancestors(root, plan.path)
        metadata = debugfs_require_path(
            root,
            plan.path,
            expected_type="regular",
        )
        size = _metadata_integer(
            metadata, "size", plan.path, 1, MAX_TARGET_BYTES
        )
        if size != plan.source_bytes:
            raise BuilderError(
                f"{plan.path} is {size} bytes, expected {plan.source_bytes}"
            )
        data = debugfs_cat(root, plan.path)
        root.assert_unchanged()
    if len(data) != size:
        raise BuilderError(f"patch target changed while reading: {plan.path}")
    return data, metadata


def _require_target_ancestors(root: OpenedRegular, target: str) -> None:
    parts = PurePosixPath(target).parts
    current = ""
    for component in parts[1:-1]:
        current += "/" + component
        debugfs_require_path(root, current, expected_type="directory")


def _metadata_integer(
    metadata: dict[str, object],
    key: str,
    path: str,
    minimum: int,
    maximum: int,
) -> int:
    value = metadata.get(key)
    if type(value) is not int or not minimum <= value <= maximum:
        raise BuilderError(f"invalid {key} metadata for patch target {path}")
    return value


def _transform_target(
    data: bytes, plan: _TargetPlan
) -> tuple[bytearray, str, tuple[_HunkState, ...]]:
    desired = bytearray(data)
    states: list[_HunkState] = []
    for hunk in plan.hunks:
        start = hunk.offset
        end = start + len(hunk.before)
        actual = data[start:end]
        if actual == hunk.before:
            state = "stock"
        elif actual == hunk.after:
            state = "replacement"
            desired[start:end] = hunk.before
        else:
            raise BuilderError(
                f"{plan.path} has unrecognized bytes at offset 0x{start:x}"
            )
        states.append(
            _HunkState(
                patch_id=hunk.patch_id,
                index=hunk.index,
                offset=start,
                size=len(hunk.before),
                state=state,
            )
        )

    reconstructed_sha256 = _sha256(desired)
    if reconstructed_sha256 != plan.source_sha256:
        raise BuilderError(
            f"{plan.path} does not reconstruct to the supported source fingerprint"
        )
    for hunk in plan.hunks:
        start = hunk.offset
        desired[start : start + len(hunk.after)] = hunk.after
    return desired, reconstructed_sha256, tuple(states)


def _replace_target(
    root_path: Path,
    target: str,
    desired: bytearray,
    *,
    mode: int,
    uid: int,
    gid: int,
) -> None:
    with tempfile.TemporaryFile() as payload:
        view = memoryview(desired)
        position = 0
        while position < len(view):
            written = os.write(payload.fileno(), view[position : position + 1024 * 1024])
            if written <= 0:
                raise BuilderError(f"short staged patch write for {target}")
            position += written
        payload.flush()
        os.fsync(payload.fileno())
        os.lseek(payload.fileno(), 0, os.SEEK_SET)
        debugfs_replace_regular(
            root_path,
            target,
            payload.fileno(),
            mode=mode,
            uid=uid,
            gid=gid,
        )
        view.release()


def _verify_target_after(
    root_path: Path, plan: _TargetPlan, expected: _TargetSnapshot
) -> None:
    current = _inspect_target(root_path, plan)
    if (
        current.current_sha256 != expected.desired_sha256
        or current.desired_sha256 != expected.desired_sha256
        or any(state.state != "replacement" for state in current.hunk_states)
        or current.mode != expected.mode
        or current.uid != expected.uid
        or current.gid != expected.gid
    ):
        raise BuilderError(f"installed patch verification failed for {plan.path}")


def _build_reports(
    patches: Sequence[PatchRecipe],
    before: dict[str, _TargetSnapshot],
    after: dict[str, _TargetSnapshot],
) -> list[dict[str, object]]:
    reports: list[dict[str, object]] = []
    for recipe in patches:
        target_reports: list[dict[str, object]] = []
        applied_hunk_bytes = 0
        for target in recipe.targets:
            before_target = before[target.path]
            after_target = after[target.path]
            states = {
                (item.patch_id, item.index): item
                for item in before_target.hunk_states
            }
            hunk_reports: list[dict[str, object]] = []
            target_states: list[str] = []
            for hunk in target.hunks:
                state = states[(recipe.patch_id, hunk.index)]
                changed = state.state == "stock"
                if changed:
                    applied_hunk_bytes += state.size
                target_states.append(state.state)
                hunk_reports.append(
                    {
                        "index": hunk.index,
                        "offset": hunk.offset,
                        "bytes": len(hunk.before),
                        "state_before": state.state,
                        "changed": changed,
                    }
                )
            if all(value == "stock" for value in target_states):
                state_before = "stock"
            elif all(value == "replacement" for value in target_states):
                state_before = "replacement"
            else:
                state_before = "partial"
            target_reports.append(
                {
                    "path": target.path,
                    "bytes": target.source_bytes,
                    "source_sha256": target.source_sha256,
                    "sha256_before": before_target.current_sha256,
                    "sha256_after": after_target.current_sha256,
                    "state_before": state_before,
                    "state_after": "replacement",
                    "hunks": hunk_reports,
                }
            )
        report: dict[str, object] = {
            "schema": PATCH_REPORT_SCHEMA,
            "id": recipe.patch_id,
            "definition_bytes": recipe.file_bytes,
            "definition_sha256": recipe.definition_sha256,
            "declared_hunk_bytes": recipe.changed_bytes,
            "applied_hunk_bytes": applied_hunk_bytes,
            "targets": target_reports,
        }
        if recipe.revision is not None:
            report["revision"] = recipe.revision
        reports.append(report)
    return reports


def _sha256(value: bytes | bytearray) -> str:
    return hashlib.sha256(value).hexdigest()
