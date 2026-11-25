#!/usr/bin/env python3
"""Read-only integrity check for the official VITON-HD folder layout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from PIL import Image


ASSETS = {
    "person": lambda split, person, garment: split / "image" / person,
    "garment": lambda split, person, garment: split / "cloth" / garment,
    "garment_mask": lambda split, person, garment: split / "cloth-mask" / garment,
    "schp_parsing": lambda split, person, garment: split / "image-parse-v3" / f"{Path(person).stem}.png",
    "densepose": lambda split, person, garment: split / "image-densepose" / f"{Path(person).stem}.jpg",
    "openpose_render": lambda split, person, garment: split / "openpose_img" / f"{Path(person).stem}_rendered.png",
    "openpose_json": lambda split, person, garment: split / "openpose_json" / f"{Path(person).stem}_keypoints.json",
}


def read_pairs(path: Path) -> tuple[list[tuple[str, str]], list[str]]:
    pairs: list[tuple[str, str]] = []
    errors: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [], [f"cannot read pair list {path}: {exc}"]
    for line_no, line in enumerate(lines, 1):
        fields = line.split()
        if len(fields) != 2:
            errors.append(f"{path}:{line_no}: expected person and garment IDs, got {line!r}")
            continue
        pairs.append((fields[0], fields[1]))
    if len(set(pairs)) != len(pairs):
        errors.append(f"{path}: duplicate person/garment pairs found")
    if len({person for person, _ in pairs}) != len(pairs):
        errors.append(f"{path}: duplicate person IDs found")
    return pairs, errors


def validate_split(root: Path, name: str, expected_hw: tuple[int, int]) -> dict:
    split_dir = root / name
    pairs, errors = read_pairs(root / f"{name}_pairs.txt")
    expected_persons = {person for person, _ in pairs}
    image_dir = split_dir / "image"
    actual_persons = {p.name for p in image_dir.glob("*") if p.is_file()} if image_dir.is_dir() else set()
    missing_person_images = sorted(expected_persons - actual_persons)
    extra_person_images = sorted(actual_persons - expected_persons)
    if missing_person_images:
        errors.append(f"{name}: {len(missing_person_images)} pair-listed person image(s) missing; first={missing_person_images[:3]}")
    if extra_person_images:
        errors.append(f"{name}: {len(extra_person_images)} unlisted person image(s); first={extra_person_images[:3]}")

    checked_files = 0
    image_dimensions: dict[str, int] = {}
    for person, garment in pairs:
        for kind, resolver in ASSETS.items():
            path = resolver(split_dir, person, garment)
            if not path.is_file():
                errors.append(f"{name}: missing {kind} for {person} / {garment}: {path}")
                continue
            if kind == "openpose_json":
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    people = payload.get("people")
                    if not people or len(people[0].get("pose_keypoints_2d", [])) < 51:
                        errors.append(f"{name}: invalid/empty OpenPose keypoints for {person}: {path}")
                except (OSError, ValueError, TypeError) as exc:
                    errors.append(f"{name}: unreadable OpenPose JSON for {person}: {path}: {exc}")
                checked_files += 1
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    height, width = image.height, image.width
                    observed = (height, width)
                    image_dimensions[f"{width}x{height}"] = image_dimensions.get(f"{width}x{height}", 0) + 1
                if observed != expected_hw:
                    errors.append(f"{name}: {kind} dimensions for {person} are {observed}, expected HxW={expected_hw}: {path}")
            except (OSError, ValueError) as exc:
                errors.append(f"{name}: unreadable {kind} for {person}: {path}: {exc}")
            checked_files += 1

    return {
        "split": name,
        "pair_count": len(pairs),
        "unique_person_count": len(expected_persons),
        "unique_garment_count": len({garment for _, garment in pairs}),
        "person_image_count": len(actual_persons),
        "checked_files": checked_files,
        "image_dimensions": image_dimensions,
        "errors": errors,
    }


def validate_dataset(root: Path, splits: Iterable[str], expected_hw: tuple[int, int]) -> list[dict]:
    return [validate_split(root, name, expected_hw) for name in splits]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/VTON-HD"))
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=768)
    args = parser.parse_args()
    splits = ("train", "test") if args.split == "all" else (args.split,)
    reports = validate_dataset(args.root, splits, (args.height, args.width))
    failures = sum(len(report["errors"]) for report in reports)
    for report in reports:
        print(json.dumps(report, sort_keys=True))
    print(f"SUMMARY root={args.root} splits={','.join(splits)} errors={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
