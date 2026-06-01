#!/usr/bin/env python3
"""Cut three fixed-length continuous test snippets per video."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import shutil

from common import ROOT, read_json, write_json


SOURCE_URLS = {
    "P03_120": "https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m/P03/videos/P03_120.MP4",
    "P06_108": "https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m/P06/videos/P06_108.MP4",
    "P22_107": "https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m/P22/videos/P22_107.MP4",
}

FRAME_TAR_URLS = {
    "P08_17": "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/rgb/test/P08/P08_17.tar",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/epic_kitchens/video_snippets/test_set/manifest.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/epic_kitchens/video_snippets/test_set/continuous_segments")
    parser.add_argument("--source-fps", type=float, default=50.0)
    parser.add_argument("--duration-seconds", type=float, default=30.0)
    parser.add_argument("--snippets-per-video", type=int, default=3)
    parser.add_argument("--reencode", action="store_true", help="Use slower frame-accurate H.264 re-encoding instead of fast stream copy.")
    parser.add_argument("--video-id", action="append", help="Limit to one or more video ids.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    selected = set(args.video_id or [])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    snippet_frames = int(round(args.duration_seconds * args.source_fps))
    output = {
        "source_manifest": relative(args.manifest),
        "source_fps": args.source_fps,
        "duration_seconds": args.duration_seconds,
        "snippet_frames": snippet_frames,
        "snippets_per_video": args.snippets_per_video,
        "selection": "centers are even quantiles of each video's annotated test-frame list",
        "cut_mode": "reencode" if args.reencode else "stream_copy",
        "videos": {},
    }

    for video_id, details in manifest.get("videos", {}).items():
        if selected and video_id not in selected:
            continue
        frames = [int(frame) for frame in details["frame_indices"]]
        snippets = choose_snippets(frames, args.snippets_per_video, snippet_frames)
        source_url = SOURCE_URLS.get(video_id)
        tar_url = FRAME_TAR_URLS.get(video_id)
        if source_url is None:
            records = make_tar_snippets(video_id, snippets, tar_url, args) if tar_url else []
            output["videos"][video_id] = {"status": "ready" if records else "source_unavailable", "source_url": tar_url, "snippets": records}
            continue

        records = []
        for item in snippets:
            out = snippet_path(args.output_dir, video_id, item["index"], item["start_frame"], item["end_frame"])
            status = "planned"
            if not args.dry_run:
                if out.exists() and not args.overwrite:
                    status = "ready"
                else:
                    cut_remote_segment(source_url, out, item["start_frame"], item["end_frame"], args.source_fps, args.reencode)
                    status = "ready"
            records.append(snippet_record(args.output_dir, video_id, item, source_url, status, args.source_fps))
            print(f"{video_id} snippet {item['index']:02d}: {status} {item['start_frame']}-{item['end_frame']} -> {out}")

        output["videos"][video_id] = {"status": "ready", "source_url": source_url, "snippets": records}

    write_json(args.output_dir / "manifest.json", output)


def choose_snippets(frames: list[int], count: int, length: int) -> list[dict]:
    snippets = []
    half = length // 2
    for index in range(count):
        rank = round((index + 0.5) * len(frames) / count - 0.5)
        rank = max(0, min(len(frames) - 1, rank))
        center = frames[rank]
        start = max(0, center - half)
        end = start + length - 1
        annotated = [frame for frame in frames if start <= frame <= end]
        snippets.append(
            {
                "index": index + 1,
                "center_frame": center,
                "start_frame": start,
                "end_frame": end,
                "annotated_frames": annotated,
            }
        )
    return snippets


def snippet_record(output_dir: Path, video_id: str, item: dict, source_url: str | None, status: str, fps: float) -> dict:
    out = snippet_path(output_dir, video_id, item["index"], item["start_frame"], item["end_frame"])
    return {
        "index": item["index"],
        "status": status,
        "source_url": source_url,
        "start_frame": item["start_frame"],
        "end_frame": item["end_frame"],
        "center_frame": item["center_frame"],
        "duration_s": round((item["end_frame"] - item["start_frame"] + 1) / fps, 3),
        "annotated_frame_count": len(item["annotated_frames"]),
        "annotated_frames": item["annotated_frames"],
        "video": relative(out),
    }


def snippet_path(output_dir: Path, video_id: str, index: int, start: int, end: int) -> Path:
    name = f"{video_id}_continuous_{index:02d}_frames_{start:07d}_{end:07d}.mp4"
    return output_dir / name


def make_tar_snippets(video_id: str, snippets: list[dict], tar_url: str | None, args: argparse.Namespace) -> list[dict]:
    if tar_url is None:
        return []
    missing = [item for item in snippets if args.overwrite or not snippet_path(args.output_dir, video_id, item["index"], item["start_frame"], item["end_frame"]).exists()]
    if not args.dry_run and missing:
        frame_dir = args.output_dir / f".{video_id}_frames_tmp"
        if frame_dir.exists():
            shutil.rmtree(frame_dir)
        frame_dir.mkdir(parents=True)
        try:
            extract_tar_frames(tar_url, frame_dir, sorted({frame for item in missing for frame in range(item["start_frame"], item["end_frame"] + 1)}))
            for item in missing:
                encode_frame_snippet(frame_dir, snippet_path(args.output_dir, video_id, item["index"], item["start_frame"], item["end_frame"]), item["start_frame"], item["end_frame"], args.source_fps)
        finally:
            shutil.rmtree(frame_dir, ignore_errors=True)

    records = []
    for item in snippets:
        out = snippet_path(args.output_dir, video_id, item["index"], item["start_frame"], item["end_frame"])
        status = "planned" if args.dry_run else ("ready" if out.exists() else "not_created")
        records.append(snippet_record(args.output_dir, video_id, item, tar_url, status, args.source_fps))
        print(f"{video_id} snippet {item['index']:02d}: {status} {item['start_frame']}-{item['end_frame']} -> {out}")
    return records


def extract_tar_frames(tar_url: str, frame_dir: Path, frames: list[int]) -> None:
    list_path = frame_dir / "frames_to_extract.txt"
    list_path.write_text("".join(f"./frame_{frame:010d}.jpg\n" for frame in frames), encoding="utf-8")
    curl = subprocess.Popen(["curl", "-L", "--fail", tar_url], stdout=subprocess.PIPE)
    try:
        tar = subprocess.run(["tar", "-xf", "-", "-C", str(frame_dir), "-T", str(list_path)], stdin=curl.stdout, check=False)
    finally:
        if curl.stdout is not None:
            curl.stdout.close()
    curl_status = curl.wait()
    if curl_status != 0 or tar.returncode != 0:
        raise RuntimeError(f"Could not extract selected frames from {tar_url}")


def encode_frame_snippet(frame_dir: Path, output: Path, start: int, end: int, fps: float) -> None:
    missing = [frame for frame in range(start, end + 1) if not (frame_dir / f"frame_{frame:010d}.jpg").exists()]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} extracted frames for {output.name}; first missing frame {missing[0]}")
    tmp = output.with_name(output.stem + ".tmp.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-start_number",
            str(start),
            "-i",
            str(frame_dir / "frame_%010d.jpg"),
            "-frames:v",
            str(end - start + 1),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(tmp),
        ],
        check=True,
    )
    probe_video(tmp)
    tmp.replace(output)


def cut_remote_segment(source_url: str, output: Path, start: int, end: int, fps: float, reencode: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.stem + ".tmp.mp4")
    duration_s = (end - start + 1) / fps
    command = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start / fps:.3f}", "-i", source_url, "-t", f"{duration_s:.3f}", "-map", "0:v:0", "-an"]
    if reencode:
        command += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]
    else:
        command += ["-c:v", "copy", "-avoid_negative_ts", "make_zero"]
    command += ["-movflags", "+faststart", str(tmp)]
    subprocess.run(command, check=True)
    probe_video(tmp)
    tmp.replace(output)


def probe_video(path: Path) -> None:
    subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


if __name__ == "__main__":
    main()
