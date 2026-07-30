from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from . import __version__
from .errors import BuilderError
from .update_inputs import build_image_from_update


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image-builder",
        description=(
            "Build a 120 GB SSD image for Genesis G80 DH 9.2-inch "
            "head units from an owner-supplied navigation update. Inputs are "
            "validated automatically; block devices are never written."
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build",
        help="assemble a new sparse regular-file SSD image",
        description=(
            "Provide the official navigation update package directory. "
            "The extracted update key is loaded from image_builder/resources."
        ),
    )
    build.add_argument(
        "--update-dir",
        required=True,
        help="directory containing the encrypted official update files",
    )
    patch_selection = build.add_mutually_exclusive_group()
    patch_selection.add_argument(
        "--no-patches",
        action="store_true",
        help="build without locally populated byte patches",
    )
    patch_selection.add_argument(
        "--exclude-patch",
        action="append",
        default=[],
        metavar="ID",
        help="omit one local byte patch; repeat as needed",
    )
    feature_selection = build.add_mutually_exclusive_group()
    feature_selection.add_argument(
        "--no-features",
        action="store_true",
        help="build without locally populated feature packages",
    )
    feature_selection.add_argument(
        "--exclude-feature",
        action="append",
        default=[],
        metavar="ID",
        help="omit one local feature package; repeat as needed",
    )
    build.add_argument("--output", required=True)
    build.add_argument("--json", action="store_true")
    return parser


def _progress(message: str) -> None:
    print(f"[image-builder] {message}", file=sys.stderr, flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        value = build_image_from_update(
            update_directory=args.update_dir,
            output_path=args.output,
            apply_default_patches=not args.no_patches,
            excluded_patch_ids=args.exclude_patch,
            apply_default_features=not args.no_features,
            excluded_feature_ids=args.exclude_feature,
            progress=_progress,
        )
        _emit(value, args.json)
        return 0
    except BuilderError as exc:
        print(f"image-builder: error: {exc}", file=sys.stderr)
        return 2


def _emit(value: dict[str, object], as_json: bool) -> None:
    if as_json:
        print(json.dumps(value, indent=2, sort_keys=True))
        return
    print(f"built {value['output']}")
    print(
        "verified source partitions before modifications and image "
        f"partitions afterwards; disk ID: {value['disk_id']}"
    )
    if "update_verification" in value:
        print("update package: shipped integrity values verified")
    feature_install = value.get("feature_install")
    if isinstance(feature_install, dict):
        features = feature_install.get("features", [])
        if features:
            print(
                "feature packages: "
                + ", ".join(
                    "%s@%s" % (item["id"], item["revision"])
                    for item in features
                )
            )
        public_patches = feature_install.get("resolved_public_patches", [])
        if public_patches:
            print("byte patches: " + ", ".join(public_patches))
