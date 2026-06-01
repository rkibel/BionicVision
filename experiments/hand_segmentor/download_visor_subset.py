#!/usr/bin/env python3
"""Download the VISOR sparse RGB/dense annotation subset for hand segmentation."""

from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path

from common import DATA_ROOT, SPLITS, write_json

BASE = "https://data.bris.ac.uk/datasets/2v6cgv1x04ol22qp9rm9x2j6a7"
VAL_SOURCE_VIDEOS = {
    "P03_22",
    "P03_14",
    "P04_06",
    "P04_24",
    "P06_03",
    "P06_10",
    "P07_101",
    "P07_103",
    "P07_110",
    "P09_02",
    "P09_07",
    "P09_103",
    "P09_104",
    "P09_106",
    "P02_02",
    "P21_01",
    "P25_09",
    "P26_01",
    "P26_02",
    "P28_05",
    "P29_04",
    "P30_07",
    "P30_110",
    "P32_07",
    "P37_102",
    "P06_106",
    "P12_04",
    "P25_101",
    "P26_108",
    "P03_120",
    "P06_108",
    "P08_17",
    "P22_107",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    selected_splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    videos = [video_id for split in selected_splits for video_id in SPLITS[split]]
    for video_id in videos:
        visor_split = "val" if video_id in VAL_SOURCE_VIDEOS else "train"
        download_video(video_id, visor_split=visor_split, force=args.force)
    write_json(
        DATA_ROOT / "subset_manifest.json",
        {"base_url": BASE, "splits": {split: list(SPLITS[split]) for split in selected_splits}},
    )


def download_video(video_id: str, *, visor_split: str, force: bool) -> None:
    participant = video_id.split("_", maxsplit=1)[0]
    rgb_url = f"{BASE}/GroundTruth-SparseAnnotations/rgb_frames/{visor_split}/{participant}/{video_id}.zip"
    dense_url = f"{BASE}/Interpolations-DenseAnnotations/{visor_split}/{video_id}_interpolations.zip"

    rgb_zip = DATA_ROOT / "sparse_rgb_zips" / f"{video_id}.zip"
    dense_zip = DATA_ROOT / "dense_annotation_zips" / f"{video_id}_interpolations.zip"
    rgb_dir = DATA_ROOT / "sparse_rgb_frames" / video_id
    dense_dir = DATA_ROOT / "dense_annotations" / video_id
    rgb_expected = rgb_dir / f"{video_id}_frame_"
    dense_expected = dense_dir / f"{video_id}_interpolations.json"

    if expected_exists(rgb_expected) and expected_exists(dense_expected) and not force:
        print(f"ready {video_id} ({visor_split})")
        return
    download(rgb_url, rgb_zip, force=force)
    download(dense_url, dense_zip, force=force)
    extract_zip(rgb_zip, rgb_dir, expected=rgb_expected, force=force)
    extract_zip(dense_zip, dense_dir, expected=dense_expected, force=force)
    rgb_zip.unlink(missing_ok=True)
    dense_zip.unlink(missing_ok=True)
    print(f"ready {video_id} ({visor_split})")


def download(url: str, output: Path, *, force: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 0 and not force:
        return
    print(f"download {url}")
    tmp = output.with_suffix(output.suffix + ".tmp")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(output)


def extract_zip(path: Path, output_dir: Path, *, expected: Path, force: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and expected_exists(expected) and not force:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        archive.extractall(output_dir)


def expected_exists(path: Path) -> bool:
    if str(path).endswith("_frame_"):
        return any(path.parent.glob(path.name + "*.jpg"))
    return path.exists()


if __name__ == "__main__":
    main()
