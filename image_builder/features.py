from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Sequence

from . import extfs
from . import patches as patch_engine
from .errors import BuilderError
from .io_utils import OpenedRegular, open_regular_read
from .patches import PatchRecipe, apply_external_patches


FEATURE_SCHEMA = "dh-headunit-feature/v1"
FEATURE_REPORT_SCHEMA = "dh-headunit-feature-install-report/v1"
MAX_FEATURE_PACKAGES = 32
MAX_FEATURE_COMPONENTS = 64
MAX_FEATURE_VARIANTS = 16
MAX_FEATURE_MANIFEST_BYTES = 256 * 1024
MAX_FEATURE_PAYLOAD_BYTES = 64 * 1024 * 1024
MAX_FEATURE_PAYLOAD_FILES = 64

_ID_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_PATH_COMPONENT_RE = re.compile(r"[A-Za-z0-9._+-]+\Z")
_MODE_RE = re.compile(r"0[0-7]{3}\Z")
_PRINTABLE_ASCII_RE = re.compile(r"[ -~]+\Z")
Progress = Callable[[str], None]


@dataclass(frozen=True)
class FeatureFile:
    feature_id: str
    component_id: str
    target: str
    payload_member: str
    payload_data: bytes = field(repr=False)
    payload_sha256: str
    mode: int
    uid: int
    gid: int


@dataclass(frozen=True)
class FeatureFixedPatch:
    component_id: str
    recipe: PatchRecipe


@dataclass(frozen=True)
class FeaturePatchVariant:
    variant_id: str
    recipe: PatchRecipe


@dataclass(frozen=True)
class FeatureVariantPatch:
    component_id: str
    variants: tuple[FeaturePatchVariant, ...]


FeatureComponent = FeatureFile | FeatureFixedPatch | FeatureVariantPatch


@dataclass(frozen=True)
class FeatureDefinition:
    feature_id: str
    revision: int
    manifest_sha256: str
    requires_features: tuple[str, ...]
    requires_byte_patches: tuple[str, ...]
    conflicts: tuple[str, ...]
    components: tuple[FeatureComponent, ...]


@dataclass(frozen=True)
class FeatureCatalog:
    catalog_root: Path
    public_patches: tuple[PatchRecipe, ...]
    features: tuple[FeatureDefinition, ...]


@dataclass(frozen=True)
class PlannedFeaturePatch:
    component_id: str
    patch_id: str
    variant_id: str | None


@dataclass(frozen=True)
class PlannedFeature:
    definition: FeatureDefinition
    files: tuple[FeatureFile, ...]
    patches: tuple[PlannedFeaturePatch, ...]


@dataclass(frozen=True)
class FeatureInstallPlan:
    catalog_root: Path
    requested_feature_ids: tuple[str, ...]
    requested_public_patch_ids: tuple[str, ...]
    excluded_feature_ids: tuple[str, ...]
    excluded_public_patch_ids: tuple[str, ...]
    resolved_public_patches: tuple[PatchRecipe, ...]
    features: tuple[PlannedFeature, ...]
    files: tuple[FeatureFile, ...]
    patches: tuple[PatchRecipe, ...]


def load_feature_catalog(
    catalog_root: str | os.PathLike[str],
) -> FeatureCatalog:
    """Load and fully validate the image builder's local drop-in catalog.

    The directory must contain ``byte_patches`` and ``feature_packages``.
    Their tracked ``.gitignore`` placeholders are ignored. Payload bytes are
    captured while the catalog is loaded so a later build uses the exact data
    authenticated by each feature manifest.
    """

    root = _require_directory(Path(catalog_root), "catalog root")
    patch_directory = _require_directory(root / "byte_patches", "byte-patch directory")
    feature_directory = _require_directory(
        root / "feature_packages", "feature-package directory"
    )
    public_patches = patch_engine.load_patch_catalog(patch_directory)

    try:
        feature_names = sorted(
            name for name in os.listdir(feature_directory) if name != ".gitignore"
        )
    except OSError as exc:
        raise BuilderError(
            f"cannot list feature-package directory {feature_directory}: "
            f"{exc.strerror}"
        ) from exc
    if len(feature_names) > MAX_FEATURE_PACKAGES:
        raise BuilderError("feature-package directory contains more than 32 entries")

    all_patch_ids = {recipe.patch_id for recipe in public_patches}
    all_payload_targets: set[str] = set()
    features: list[FeatureDefinition] = []
    total_manifest_bytes = 0
    total_payload_bytes = 0
    total_payload_files = 0
    hidden_patch_count = 0
    variant_patch_count = 0

    for feature_id in feature_names:
        if _ID_RE.fullmatch(feature_id) is None:
            raise BuilderError(f"invalid feature-package directory: {feature_id}")
        package_root = _require_directory(
            feature_directory / feature_id,
            f"feature package {feature_id}",
        )
        manifest, manifest_bytes, manifest_sha256 = _read_manifest(
            package_root / "feature.json", feature_id
        )
        total_manifest_bytes += manifest_bytes
        if total_manifest_bytes > MAX_FEATURE_MANIFEST_BYTES * MAX_FEATURE_PACKAGES:
            raise BuilderError("feature manifests exceed aggregate size limit")
        feature, expected_files = _parse_feature(
            package_root,
            feature_id,
            manifest,
            manifest_sha256,
            all_patch_ids,
        )
        _validate_package_contents(package_root, expected_files, feature_id)

        for component in feature.components:
            if isinstance(component, FeatureFile):
                if component.target in all_payload_targets:
                    raise BuilderError(
                        f"multiple features own payload target {component.target}"
                    )
                all_payload_targets.add(component.target)
                total_payload_bytes += len(component.payload_data)
                total_payload_files += 1
            elif isinstance(component, FeatureFixedPatch):
                hidden_patch_count += 1
            elif isinstance(component, FeatureVariantPatch):
                variant_patch_count += len(component.variants)
        if total_payload_bytes > MAX_FEATURE_PAYLOAD_BYTES:
            raise BuilderError("feature payloads exceed 64 MiB")
        if total_payload_files > MAX_FEATURE_PAYLOAD_FILES:
            raise BuilderError("feature packages contain too many payload files")
        features.append(feature)

    if len(public_patches) + hidden_patch_count + variant_patch_count > patch_engine.MAX_PATCH_FILES:
        raise BuilderError("combined patch catalog contains more than 128 patches")
    _validate_feature_graph(features, public_patches)
    _validate_payload_patch_disjoint(features, public_patches)
    _validate_definition_limits(features, public_patches)
    return FeatureCatalog(
        catalog_root=root,
        public_patches=public_patches,
        features=tuple(features),
    )


def plan_feature_install(
    root_path: str | os.PathLike[str],
    catalog: FeatureCatalog,
    *,
    selected_feature_ids: Iterable[str],
    selected_public_patch_ids: Iterable[str] = (),
    excluded_feature_ids: Iterable[str] = (),
    excluded_public_patch_ids: Iterable[str] = (),
) -> FeatureInstallPlan:
    """Resolve an explicit install selection against one restored p2 image.

    Feature dependencies and required public byte patches are selected
    automatically.  Regional ``byte-patch-one-of`` components are selected
    solely by exact target preimage compatibility; callers cannot force a
    region variant.
    """

    if not isinstance(catalog, FeatureCatalog):
        raise BuilderError("feature catalog is invalid")
    root = Path(root_path)
    feature_by_id = {feature.feature_id: feature for feature in catalog.features}
    public_by_id = {recipe.patch_id: recipe for recipe in catalog.public_patches}
    requested_features = _normalize_selection(
        selected_feature_ids,
        feature_by_id,
        "feature",
    )
    requested_public = _normalize_selection(
        selected_public_patch_ids,
        public_by_id,
        "public patch",
    )
    excluded_features = _normalize_selection(
        excluded_feature_ids,
        feature_by_id,
        "excluded feature",
    )
    excluded_public = _normalize_selection(
        excluded_public_patch_ids,
        public_by_id,
        "excluded public patch",
    )
    selected_and_excluded_features = sorted(
        set(requested_features) & set(excluded_features)
    )
    if selected_and_excluded_features:
        raise BuilderError(
            "feature is both selected and excluded: "
            + selected_and_excluded_features[0]
        )
    selected_and_excluded_public = sorted(
        set(requested_public) & set(excluded_public)
    )
    if selected_and_excluded_public:
        raise BuilderError(
            "public patch is both selected and excluded: "
            + selected_and_excluded_public[0]
        )
    if not requested_features and not requested_public:
        if catalog.features or catalog.public_patches:
            raise BuilderError("feature and public-patch selection is empty")
        return FeatureInstallPlan(
            catalog_root=catalog.catalog_root,
            requested_feature_ids=(),
            requested_public_patch_ids=(),
            excluded_feature_ids=excluded_features,
            excluded_public_patch_ids=excluded_public,
            resolved_public_patches=(),
            features=(),
            files=(),
            patches=(),
        )

    ordered_features = _resolve_feature_dependencies(requested_features, feature_by_id)
    resolved_feature_ids = {feature.feature_id for feature in ordered_features}
    required_excluded_features = sorted(resolved_feature_ids & set(excluded_features))
    if required_excluded_features:
        raise BuilderError(
            "enabled feature requires excluded feature: "
            + required_excluded_features[0]
        )
    _check_selected_conflicts(ordered_features, resolved_feature_ids)

    resolved_public_ids = set(requested_public)
    for feature in ordered_features:
        resolved_public_ids.update(feature.requires_byte_patches)
    required_excluded_public = sorted(resolved_public_ids & set(excluded_public))
    if required_excluded_public:
        raise BuilderError(
            "enabled feature requires excluded public patch: "
            + required_excluded_public[0]
        )
    resolved_public = tuple(public_by_id[item] for item in sorted(resolved_public_ids))
    _require_offline_feature_targets(ordered_features)

    variants = _resolve_all_variants(root, catalog)
    planned_features: list[PlannedFeature] = []
    files: list[FeatureFile] = []
    hidden_recipes: list[PatchRecipe] = []
    for feature in ordered_features:
        feature_files: list[FeatureFile] = []
        feature_patches: list[PlannedFeaturePatch] = []
        for component in feature.components:
            if isinstance(component, FeatureFile):
                feature_files.append(component)
                files.append(component)
            elif isinstance(component, FeatureFixedPatch):
                hidden_recipes.append(component.recipe)
                feature_patches.append(
                    PlannedFeaturePatch(
                        component_id=component.component_id,
                        patch_id=component.recipe.patch_id,
                        variant_id=None,
                    )
                )
            elif isinstance(component, FeatureVariantPatch):
                chosen = variants.get((feature.feature_id, component.component_id))
                if chosen is None:
                    raise BuilderError(
                        f"feature {feature.feature_id} component "
                        f"{component.component_id} has no unique compatible variant"
                    )
                hidden_recipes.append(chosen.recipe)
                feature_patches.append(
                    PlannedFeaturePatch(
                        component_id=component.component_id,
                        patch_id=chosen.recipe.patch_id,
                        variant_id=chosen.variant_id,
                    )
                )
        planned_features.append(
            PlannedFeature(
                definition=feature,
                files=tuple(feature_files),
                patches=tuple(feature_patches),
            )
        )

    recipes = _sorted_recipes((*resolved_public, *hidden_recipes))
    plans = patch_engine._build_target_plans(recipes)
    _preflight_feature_files(root, files)
    _preflight_patch_plans(root, plans)
    return FeatureInstallPlan(
        catalog_root=catalog.catalog_root,
        requested_feature_ids=requested_features,
        requested_public_patch_ids=requested_public,
        excluded_feature_ids=excluded_features,
        excluded_public_patch_ids=excluded_public,
        resolved_public_patches=resolved_public,
        features=tuple(planned_features),
        files=tuple(files),
        patches=recipes,
    )


def feature_target_paths(plan: FeatureInstallPlan) -> set[str]:
    """Return paths created by the selected feature payloads."""

    _require_plan(plan)
    return {item.target for item in plan.files}


def feature_patch_target_paths(plan: FeatureInstallPlan) -> set[str]:
    """Return paths modified by the selected feature/public byte patches."""

    _require_plan(plan)
    return patch_engine.patch_target_paths(plan.patches)


def apply_feature_install(
    root_path: str | os.PathLike[str],
    plan: FeatureInstallPlan,
    *,
    progress: Progress | None = None,
) -> dict[str, object]:
    """Create selected feature payloads, then apply their combined patches.

    The caller must provide a disposable private root image.  All targets are
    rechecked before the first write, payload files are installed before loader
    hooks, and every installed file is verified after the byte patches finish.
    """

    _require_plan(plan)
    root = Path(root_path)
    patch_plans = patch_engine._build_target_plans(plan.patches)
    _preflight_feature_files(root, plan.files)
    _preflight_patch_plans(root, patch_plans)

    for item in plan.files:
        if progress is not None:
            progress(f"creating feature payload {item.target}")
        _create_feature_file(root, item)
        _verify_feature_file(root, item)

    patch_reports = apply_external_patches(
        root,
        plan.patches,
        progress=progress,
    )
    if not plan.patches and plan.files:
        if progress is not None:
            progress("checking root filesystem after feature payload creation")
        extfs.run_e2fsck_path(root, "feature-customized restored root")

    for item in plan.files:
        _verify_feature_file(root, item)
    return _build_install_report(plan, patch_reports)


def _parse_feature(
    package_root: Path,
    feature_id: str,
    manifest: object,
    manifest_sha256: str,
    all_patch_ids: set[str],
) -> tuple[FeatureDefinition, set[str]]:
    location = f"feature {feature_id}"
    _json_object(
        manifest,
        location,
        (
            "schema",
            "revision",
            "requires_features",
            "requires_byte_patches",
            "conflicts",
            "components",
        ),
        optional=("_comment",),
    )
    assert isinstance(manifest, dict)
    if manifest["schema"] != FEATURE_SCHEMA:
        raise BuilderError(f"{location} uses an unsupported schema")
    revision = _json_integer(manifest["revision"], f"{location} revision", 1, 0x7FFFFFFF)
    requires_features = _id_list(
        manifest["requires_features"], f"{location} requires_features"
    )
    requires_patches = _id_list(
        manifest["requires_byte_patches"],
        f"{location} requires_byte_patches",
    )
    conflicts = _id_list(manifest["conflicts"], f"{location} conflicts")
    component_values = _json_list(
        manifest["components"], f"{location} components", MAX_FEATURE_COMPONENTS
    )
    if not component_values:
        raise BuilderError(f"{location} has no components")

    components: list[FeatureComponent] = []
    component_ids: set[str] = set()
    expected_files = {"feature.json"}
    for index, value in enumerate(component_values):
        component_location = f"{location} component {index}"
        if not isinstance(value, dict):
            raise BuilderError(f"{component_location} must be a JSON object")
        component_type = value.get("type")
        if component_type == "file":
            component, referenced = _parse_file_component(
                package_root, feature_id, value, component_location
            )
            expected_files.add(referenced)
        elif component_type == "byte-patch":
            _json_object(
                value,
                component_location,
                ("id", "type", "path", "bytes", "sha256"),
            )
            component_id = _identifier(value["id"], f"{component_location} id")
            recipe, referenced = _load_manifest_patch(
                package_root,
                value,
                component_location,
                all_patch_ids,
            )
            component = FeatureFixedPatch(component_id=component_id, recipe=recipe)
            expected_files.add(referenced)
        elif component_type == "byte-patch-one-of":
            component, referenced = _parse_variant_component(
                package_root,
                value,
                component_location,
                all_patch_ids,
            )
            expected_files.update(referenced)
        else:
            raise BuilderError(f"{component_location} has an unsupported component type")
        if component.component_id in component_ids:
            raise BuilderError(
                f"feature {feature_id} has duplicate component {component.component_id}"
            )
        component_ids.add(component.component_id)
        components.append(component)

    return (
        FeatureDefinition(
            feature_id=feature_id,
            revision=revision,
            manifest_sha256=manifest_sha256,
            requires_features=requires_features,
            requires_byte_patches=requires_patches,
            conflicts=conflicts,
            components=tuple(components),
        ),
        expected_files,
    )


def _parse_file_component(
    package_root: Path,
    feature_id: str,
    value: dict[str, object],
    location: str,
) -> tuple[FeatureFile, str]:
    _json_object(
        value,
        location,
        (
            "id",
            "type",
            "operation",
            "target",
            "payload",
            "bytes",
            "sha256",
            "mode",
            "uid",
            "gid",
        ),
    )
    component_id = _identifier(value["id"], f"{location} id")
    if value["operation"] != "create":
        raise BuilderError(f"{location} supports only the create operation")
    target = _printable_text(value["target"], f"{location} target", 512)
    if not _safe_feature_target_path(target):
        raise BuilderError(f"{location} has an unsafe target")
    payload_member = _printable_text(value["payload"], f"{location} payload", 512)
    if not _safe_relative_path(payload_member) or not payload_member.startswith("payload/"):
        raise BuilderError(f"{location} has an invalid payload path")
    payload_path = _package_regular_path(package_root, payload_member, location)
    payload_bytes = _json_integer(
        value["bytes"], f"{location} bytes", 1, MAX_FEATURE_PAYLOAD_BYTES
    )
    payload_sha256 = _sha256_text(value["sha256"], f"{location} sha256")
    mode_text = _printable_text(value["mode"], f"{location} mode", 4)
    if _MODE_RE.fullmatch(mode_text) is None:
        raise BuilderError(f"{location} has an invalid mode")
    uid = _json_integer(value["uid"], f"{location} uid", 0, 0x7FFFFFFF)
    gid = _json_integer(value["gid"], f"{location} gid", 0, 0x7FFFFFFF)
    payload_data = _read_regular_bytes(
        payload_path,
        maximum=MAX_FEATURE_PAYLOAD_BYTES,
        description=f"{location} payload",
    )
    if len(payload_data) != payload_bytes or _sha256(payload_data) != payload_sha256:
        raise BuilderError(f"{location} payload identity does not match its manifest")
    return (
        FeatureFile(
            feature_id=feature_id,
            component_id=component_id,
            target=target,
            payload_member=payload_member,
            payload_data=payload_data,
            payload_sha256=payload_sha256,
            mode=int(mode_text, 8),
            uid=uid,
            gid=gid,
        ),
        payload_member,
    )


def _parse_variant_component(
    package_root: Path,
    value: dict[str, object],
    location: str,
    all_patch_ids: set[str],
) -> tuple[FeatureVariantPatch, set[str]]:
    _json_object(value, location, ("id", "type", "variants"))
    component_id = _identifier(value["id"], f"{location} id")
    variant_values = _json_list(
        value["variants"], f"{location} variants", MAX_FEATURE_VARIANTS
    )
    if len(variant_values) < 2:
        raise BuilderError(f"{location} requires at least two variants")
    variants: list[FeaturePatchVariant] = []
    referenced: set[str] = set()
    target_paths: tuple[str, ...] | None = None
    for index, variant_value in enumerate(variant_values):
        variant_location = f"{location} variant {index}"
        _json_object(
            variant_value,
            variant_location,
            ("id", "path", "bytes", "sha256"),
        )
        assert isinstance(variant_value, dict)
        variant_id = _identifier(variant_value["id"], f"{variant_location} id")
        recipe, relative_path = _load_manifest_patch(
            package_root,
            variant_value,
            variant_location,
            all_patch_ids,
        )
        current_paths = tuple(target.path for target in recipe.targets)
        if target_paths is None:
            target_paths = current_paths
        elif current_paths != target_paths:
            raise BuilderError(f"{location} variants must have identical target paths")
        variants.append(FeaturePatchVariant(variant_id=variant_id, recipe=recipe))
        referenced.add(relative_path)
    variant_ids = [variant.variant_id for variant in variants]
    if variant_ids != sorted(set(variant_ids)):
        raise BuilderError(f"{location} variant ids must be sorted and unique")
    return (
        FeatureVariantPatch(component_id=component_id, variants=tuple(variants)),
        referenced,
    )


def _load_manifest_patch(
    package_root: Path,
    value: dict[str, object],
    location: str,
    all_patch_ids: set[str],
) -> tuple[PatchRecipe, str]:
    relative_path = _printable_text(value["path"], f"{location} path", 256)
    if (
        not _safe_relative_path(relative_path)
        or not relative_path.startswith("patches/")
        or not relative_path.endswith(".json")
    ):
        raise BuilderError(f"{location} has an invalid byte-patch path")
    patch_id = PurePosixPath(relative_path).name.removesuffix(".json")
    if _ID_RE.fullmatch(patch_id) is None:
        raise BuilderError(f"{location} has an invalid byte-patch filename")
    if patch_id in all_patch_ids:
        raise BuilderError(f"duplicate patch id: {patch_id}")
    definition_bytes = _json_integer(
        value["bytes"], f"{location} bytes", 1, patch_engine.MAX_PATCH_BYTES
    )
    definition_sha256 = _sha256_text(value["sha256"], f"{location} sha256")
    patch_path = _package_regular_path(package_root, relative_path, location)
    recipe = patch_engine._read_patch(patch_path, patch_id)
    if (
        recipe.file_bytes != definition_bytes
        or recipe.definition_sha256 != definition_sha256
    ):
        raise BuilderError(f"{location} byte-patch identity does not match its manifest")
    all_patch_ids.add(patch_id)
    return recipe, relative_path


def _resolve_all_variants(
    root_path: Path,
    catalog: FeatureCatalog,
) -> dict[tuple[str, str], FeaturePatchVariant | None]:
    fixed = [
        component.recipe
        for feature in catalog.features
        for component in feature.components
        if isinstance(component, FeatureFixedPatch)
    ]
    base = list(catalog.public_patches) + fixed
    patch_engine._build_target_plans(_sorted_recipes(base))
    earlier: list[PatchRecipe] = []
    selected: dict[tuple[str, str], FeaturePatchVariant | None] = {}
    for feature in catalog.features:
        for component in feature.components:
            if not isinstance(component, FeatureVariantPatch):
                continue
            compatible: list[FeaturePatchVariant] = []
            for variant in component.variants:
                trial = _sorted_recipes((*base, *earlier, variant.recipe))
                try:
                    plans = patch_engine._build_target_plans(trial)
                    for target in variant.recipe.targets:
                        patch_engine._inspect_target(root_path, plans[target.path])
                except BuilderError:
                    continue
                compatible.append(variant)
            key = (feature.feature_id, component.component_id)
            if len(compatible) == 1:
                selected[key] = compatible[0]
                earlier.append(compatible[0].recipe)
            else:
                selected[key] = None
    return selected


def _resolve_feature_dependencies(
    requested: tuple[str, ...],
    feature_by_id: dict[str, FeatureDefinition],
) -> tuple[FeatureDefinition, ...]:
    ordered: list[FeatureDefinition] = []
    visited: set[str] = set()

    def visit(feature_id: str) -> None:
        if feature_id in visited:
            return
        feature = feature_by_id[feature_id]
        for required_id in feature.requires_features:
            visit(required_id)
        visited.add(feature_id)
        ordered.append(feature)

    for feature_id in requested:
        visit(feature_id)
    return tuple(ordered)


def _check_selected_conflicts(
    features: Sequence[FeatureDefinition],
    selected_ids: set[str],
) -> None:
    by_id = {feature.feature_id: feature for feature in features}
    for feature in features:
        for other_id in sorted(selected_ids):
            if other_id == feature.feature_id:
                continue
            other = by_id[other_id]
            if other_id in feature.conflicts or feature.feature_id in other.conflicts:
                first, second = sorted((feature.feature_id, other_id))
                raise BuilderError(f"features {first} and {second} conflict")


def _validate_feature_graph(
    features: Sequence[FeatureDefinition],
    public_patches: Sequence[PatchRecipe],
) -> None:
    by_id = {feature.feature_id: feature for feature in features}
    public_ids = {recipe.patch_id for recipe in public_patches}
    for feature in features:
        for patch_id in feature.requires_byte_patches:
            if patch_id not in public_ids:
                raise BuilderError(
                    f"feature {feature.feature_id} requires unknown public patch {patch_id}"
                )
        for required_id in feature.requires_features:
            if required_id == feature.feature_id or required_id not in by_id:
                raise BuilderError(
                    f"feature {feature.feature_id} requires unknown feature {required_id}"
                )
        for conflict_id in feature.conflicts:
            if conflict_id == feature.feature_id or conflict_id not in by_id:
                raise BuilderError(
                    f"feature {feature.feature_id} conflicts with unknown feature {conflict_id}"
                )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(feature_id: str) -> None:
        if feature_id in visiting:
            raise BuilderError(f"feature dependency cycle includes {feature_id}")
        if feature_id in visited:
            return
        visiting.add(feature_id)
        for required_id in by_id[feature_id].requires_features:
            visit(required_id)
        visiting.remove(feature_id)
        visited.add(feature_id)

    for feature_id in sorted(by_id):
        visit(feature_id)


def _validate_payload_patch_disjoint(
    features: Sequence[FeatureDefinition],
    public_patches: Sequence[PatchRecipe],
) -> None:
    payload_targets = {
        component.target
        for feature in features
        for component in feature.components
        if isinstance(component, FeatureFile)
    }
    recipes = list(public_patches)
    for feature in features:
        for component in feature.components:
            if isinstance(component, FeatureFixedPatch):
                recipes.append(component.recipe)
            elif isinstance(component, FeatureVariantPatch):
                recipes.extend(variant.recipe for variant in component.variants)
    patch_targets = {
        target.path for recipe in recipes for target in recipe.targets
    }
    conflicts = sorted(payload_targets & patch_targets)
    if conflicts:
        raise BuilderError(
            "feature payload target is also a byte-patch target: " + conflicts[0]
        )


def _validate_definition_limits(
    features: Sequence[FeatureDefinition],
    public_patches: Sequence[PatchRecipe],
) -> None:
    candidates = list(public_patches)
    fixed: list[PatchRecipe] = []
    for feature in features:
        for component in feature.components:
            if isinstance(component, FeatureFixedPatch):
                fixed.append(component.recipe)
            elif isinstance(component, FeatureVariantPatch):
                candidates.extend(variant.recipe for variant in component.variants)
    definitions = candidates + fixed
    if sum(recipe.file_bytes for recipe in definitions) > patch_engine.MAX_SELECTED_PATCH_BYTES:
        raise BuilderError("combined patch definitions exceed 8 MiB")
    # All fixed recipes participate in SD variant matching along with the full
    # public catalog, so structural conflicts must be rejected up front.
    patch_engine._build_target_plans(_sorted_recipes((*public_patches, *fixed)))


def _preflight_feature_files(root_path: Path, files: Sequence[FeatureFile]) -> None:
    with open_regular_read(root_path) as root:
        for item in files:
            _require_target_ancestors(root, item.target)
            extfs.debugfs_require_absent(root, item.target)
        root.assert_unchanged()


def _preflight_patch_plans(
    root_path: Path,
    plans: dict[str, object],
) -> None:
    for path in sorted(plans):
        patch_engine._inspect_target(root_path, plans[path])


def _create_feature_file(root_path: Path, item: FeatureFile) -> None:
    with tempfile.TemporaryFile() as payload:
        _write_all(payload.fileno(), item.payload_data)
        os.fsync(payload.fileno())
        os.lseek(payload.fileno(), 0, os.SEEK_SET)
        extfs.debugfs_create_regular(
            root_path,
            item.target,
            payload.fileno(),
            mode=item.mode,
            uid=item.uid,
            gid=item.gid,
        )


def _verify_feature_file(root_path: Path, item: FeatureFile) -> None:
    with open_regular_read(root_path) as root:
        metadata = extfs.debugfs_require_path(
            root,
            item.target,
            expected_type="regular",
        )
        data = extfs.debugfs_cat(root, item.target)
        root.assert_unchanged()
    if (
        data != item.payload_data
        or metadata.get("size") != len(item.payload_data)
        or metadata.get("mode") != item.mode
        or metadata.get("uid") != item.uid
        or metadata.get("gid") != item.gid
    ):
        raise BuilderError(f"installed feature payload verification failed: {item.target}")


def _build_install_report(
    plan: FeatureInstallPlan,
    patch_reports: list[dict[str, object]],
) -> dict[str, object]:
    features: list[dict[str, object]] = []
    for selected in plan.features:
        definition = selected.definition
        features.append(
            {
                "id": definition.feature_id,
                "revision": definition.revision,
                "manifest_sha256": definition.manifest_sha256,
                "requires_features": list(definition.requires_features),
                "requires_byte_patches": list(definition.requires_byte_patches),
                "selected_variants": [
                    {
                        "component_id": item.component_id,
                        "variant_id": item.variant_id,
                        "patch_id": item.patch_id,
                    }
                    for item in selected.patches
                    if item.variant_id is not None
                ],
                "component_patches": [
                    {
                        "component_id": item.component_id,
                        "patch_id": item.patch_id,
                    }
                    for item in selected.patches
                ],
                "files": [
                    {
                        "component_id": item.component_id,
                        "operation": "create",
                        "target": item.target,
                        "payload": item.payload_member,
                        "bytes": len(item.payload_data),
                        "sha256": item.payload_sha256,
                        "mode": f"{item.mode:04o}",
                        "uid": item.uid,
                        "gid": item.gid,
                    }
                    for item in selected.files
                ],
            }
        )
    return {
        "schema": FEATURE_REPORT_SCHEMA,
        "requested_features": list(plan.requested_feature_ids),
        "requested_public_patches": list(plan.requested_public_patch_ids),
        "excluded_features": list(plan.excluded_feature_ids),
        "excluded_public_patches": list(plan.excluded_public_patch_ids),
        "resolved_features": [item.definition.feature_id for item in plan.features],
        "resolved_public_patches": [
            recipe.patch_id for recipe in plan.resolved_public_patches
        ],
        "features": features,
        "external_patches": patch_reports,
    }


def _read_manifest(path: Path, feature_id: str) -> tuple[object, int, str]:
    raw = _read_regular_bytes(
        path,
        maximum=MAX_FEATURE_MANIFEST_BYTES,
        description=f"feature manifest {feature_id}",
    )
    if not raw or b"\0" in raw:
        raise BuilderError(f"feature manifest is empty or contains a NUL: {feature_id}")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except UnicodeDecodeError as exc:
        raise BuilderError(f"feature manifest is not valid UTF-8: {feature_id}") from exc
    except (TypeError, ValueError) as exc:
        raise BuilderError(f"feature manifest is invalid JSON: {feature_id}: {exc}") from exc
    return value, len(raw), _sha256(raw)


def _read_regular_bytes(path: Path, *, maximum: int, description: str) -> bytes:
    with open_regular_read(path) as opened:
        if not 1 <= opened.size <= maximum:
            raise BuilderError(f"{description} size is invalid")
        raw = os.pread(opened.fd, opened.size, 0)
        opened.assert_unchanged()
    if len(raw) != opened.size:
        raise BuilderError(f"{description} changed while being read")
    return raw


def _require_directory(path: Path, description: str) -> Path:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise BuilderError(f"cannot inspect {description} {path}: {exc.strerror}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise BuilderError(f"{description} is missing or unsafe: {path}")
    return path


def _package_regular_path(package_root: Path, relative: str, location: str) -> Path:
    if not _safe_relative_path(relative):
        raise BuilderError(f"{location} has an unsafe package path")
    current = package_root
    for component in PurePosixPath(relative).parts:
        current = current / component
        try:
            metadata = os.lstat(current)
        except OSError as exc:
            raise BuilderError(f"cannot inspect {location}: {exc.strerror}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise BuilderError(f"{location} traverses a symbolic link")
    if not stat.S_ISREG(metadata.st_mode):
        raise BuilderError(f"{location} is not a regular file")
    return current


def _validate_package_contents(
    package_root: Path,
    expected_files: set[str],
    feature_id: str,
) -> None:
    expected_directories: set[str] = set()
    for relative_file in expected_files:
        parts = PurePosixPath(relative_file).parts[:-1]
        for index in range(1, len(parts) + 1):
            expected_directories.add("/".join(parts[:index]))
    found: set[str] = set()
    for current, directories, files in os.walk(package_root, topdown=True):
        current_path = Path(current)
        relative_root = current_path.relative_to(package_root).as_posix()
        if relative_root == ".":
            relative_root = ""
        for name in list(directories):
            path = current_path / name
            relative = "/".join(part for part in (relative_root, name) if part)
            if _PATH_COMPONENT_RE.fullmatch(name) is None or path.is_symlink():
                raise BuilderError(f"feature {feature_id} contains an unsafe directory")
            if relative not in expected_directories:
                raise BuilderError(
                    f"feature {feature_id} contains unreferenced directory {relative}"
                )
        for name in files:
            path = current_path / name
            relative = "/".join(part for part in (relative_root, name) if part)
            try:
                metadata = os.lstat(path)
            except OSError as exc:
                raise BuilderError(
                    f"cannot inspect feature {feature_id} file {relative}: {exc.strerror}"
                ) from exc
            if (
                _PATH_COMPONENT_RE.fullmatch(name) is None
                or stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISREG(metadata.st_mode)
            ):
                raise BuilderError(f"feature {feature_id} contains an unsafe file")
            found.add(relative)
    unexpected = sorted(found - expected_files)
    missing = sorted(expected_files - found)
    if unexpected:
        raise BuilderError(
            f"feature {feature_id} contains unreferenced file {unexpected[0]}"
        )
    if missing:
        raise BuilderError(f"feature {feature_id} is missing referenced file {missing[0]}")


def _normalize_selection(
    values: Iterable[str],
    available: dict[str, object],
    description: str,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise BuilderError(f"selected {description} IDs must be an iterable of IDs")
    try:
        selected = list(values)
    except TypeError as exc:
        raise BuilderError(f"selected {description} IDs must be iterable") from exc
    seen: set[str] = set()
    for value in selected:
        if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
            raise BuilderError(f"selected {description} IDs must be valid IDs")
        if value in seen:
            raise BuilderError(f"duplicate selected {description}: {value}")
        if value not in available:
            choices = ", ".join(sorted(available))
            raise BuilderError(
                f"unknown selected {description} {value!r}; available: {choices}"
            )
        seen.add(value)
    return tuple(sorted(seen))


def _require_offline_feature_targets(
    features: Sequence[FeatureDefinition],
) -> None:
    for feature in features:
        for component in feature.components:
            if (
                isinstance(component, FeatureFile)
                and not patch_engine._safe_target_path(component.target)
            ):
                raise BuilderError(
                    f"feature {feature.feature_id} payload target is outside the "
                    f"restored p2 filesystem: {component.target}"
                )


def _require_target_ancestors(root: OpenedRegular, target: str) -> None:
    current = ""
    for component in PurePosixPath(target).parts[1:-1]:
        current += "/" + component
        extfs.debugfs_require_path(root, current, expected_type="directory")


def _sorted_recipes(recipes: Sequence[PatchRecipe]) -> tuple[PatchRecipe, ...]:
    values = tuple(sorted(recipes, key=lambda item: item.patch_id))
    if len({item.patch_id for item in values}) != len(values):
        raise BuilderError("selected patch recipes are not unique")
    return values


def _require_plan(plan: FeatureInstallPlan) -> None:
    if not isinstance(plan, FeatureInstallPlan):
        raise BuilderError("feature install plan is invalid")


def _json_object(
    value: object,
    location: str,
    required: Sequence[str],
    *,
    optional: Sequence[str] = (),
) -> None:
    if not isinstance(value, dict):
        raise BuilderError(f"{location} must be a JSON object")
    keys = set(value)
    required_keys = set(required)
    missing = sorted(required_keys - keys)
    unknown = sorted(keys - required_keys - set(optional))
    if missing:
        raise BuilderError(f"{location} is missing field {missing[0]}")
    if unknown:
        raise BuilderError(f"{location} has unknown field {unknown[0]}")


def _json_list(value: object, location: str, maximum: int) -> list[object]:
    if not isinstance(value, list):
        raise BuilderError(f"{location} must be a JSON array")
    if len(value) > maximum:
        raise BuilderError(f"{location} has too many entries")
    return value


def _json_integer(value: object, location: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BuilderError(f"{location} must be a JSON integer")
    if not minimum <= value <= maximum:
        raise BuilderError(f"{location} is outside the allowed range")
    return value


def _printable_text(value: object, location: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= maximum
        or _PRINTABLE_ASCII_RE.fullmatch(value) is None
    ):
        raise BuilderError(f"{location} must use printable ASCII")
    return value


def _identifier(value: object, location: str) -> str:
    text = _printable_text(value, location, 64)
    if _ID_RE.fullmatch(text) is None:
        raise BuilderError(f"{location} is not a valid identifier")
    return text


def _id_list(value: object, location: str) -> tuple[str, ...]:
    values = tuple(
        _identifier(item, f"{location}[{index}]")
        for index, item in enumerate(_json_list(value, location, 128))
    )
    if values != tuple(sorted(set(values))):
        raise BuilderError(f"{location} must be sorted and unique")
    return values


def _sha256_text(value: object, location: str) -> str:
    text = _printable_text(value, location, 64)
    if _SHA256_RE.fullmatch(text) is None:
        raise BuilderError(f"{location} has an invalid SHA-256")
    return text


def _safe_relative_path(value: str) -> bool:
    if not value or value.startswith("/") or "\\" in value:
        return False
    path = PurePosixPath(value)
    return str(path) == value and all(
        component not in ("", ".", "..")
        and _PATH_COMPONENT_RE.fullmatch(component) is not None
        for component in path.parts
    )


def _safe_feature_target_path(value: str) -> bool:
    if not value.startswith("/") or value == "/":
        return False
    return all(
        component not in ("", ".", "..")
        and _PATH_COMPONENT_RE.fullmatch(component) is not None
        for component in value.split("/")[1:]
    )


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_all(fd: int, value: bytes) -> None:
    offset = 0
    while offset < len(value):
        written = os.write(fd, value[offset:])
        if written <= 0:
            raise BuilderError("short write staging feature payload")
        offset += written
