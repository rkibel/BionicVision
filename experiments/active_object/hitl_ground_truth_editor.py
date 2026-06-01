#!/usr/bin/env python3
"""Human-in-the-loop active-object ground-truth editor using SAM clicks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

from common import ROOT, FrameKey, annotation_shape, load_frames_by_index, read_rgb, sparse_frame_path, split_frame_keys


DEFAULT_MANIFEST = ROOT / "data/epic_kitchens/HITL/active_object_prompt_frames.json"
DEFAULT_OUTPUT = ROOT / "data/epic_kitchens/HITL/active_objects"
DEFAULT_SAM = ROOT / "external/model_sources/segmentation/Tracking-Anything-with-DEVA/saves/sam_vit_h_4b8939.pth"
KEYBOARD_JS = """
() => {
  if (!window.__activeObjectEditorShortcuts) {
    window.__activeObjectEditorShortcuts = true;
    document.addEventListener("keydown", (event) => {
      const target = event.target;
      const tag = target && target.tagName ? target.tagName.toLowerCase() : "";
      const isEditable = ["input", "textarea", "select"].includes(tag) || (target && target.isContentEditable);
      if (isEditable || event.ctrlKey || event.metaKey || event.altKey) {
        return;
      }
      const key = event.key.toLowerCase();
      if (key === "a") {
        event.preventDefault();
        const button = document.querySelector("#accept-proposal-button button") || document.querySelector("#accept-proposal-button");
        if (button) button.click();
      }
      if (key === "s") {
        event.preventDefault();
        const button = document.querySelector("#submit-ground-truth-button button") || document.querySelector("#submit-ground-truth-button");
        if (button) button.click();
      }
      if (key === "e") {
        event.preventDefault();
        const button = document.querySelector("#undo-point-button button") || document.querySelector("#undo-point-button");
        if (button) button.click();
      }
    });
  }
  return [];
}
"""
EDITOR_CSS = """
html,
body,
gradio-app,
.gradio-container {
  height: 100vh !important;
  max-height: 100vh !important;
  overflow: hidden !important;
}

footer,
.footer,
#footer,
.built-with,
.api-docs,
a[href*="gradio.app"] {
  display: none !important;
}

.gradio-container {
  padding-bottom: 0 !important;
}

#editor-image {
  max-height: calc(100vh - 220px) !important;
  overflow: hidden !important;
}

#editor-image img,
#editor-image canvas {
  max-height: calc(100vh - 250px) !important;
  object-fit: contain !important;
}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sam-checkpoint", type=Path, default=DEFAULT_SAM)
    parser.add_argument("--sam-model", default="vit_h")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    prompts = load_frame_prompts(args.source_manifest, args.split)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(f"frames={len(prompts)} output_dir={args.output_dir}")
        for prompt in prompts[:10]:
            print(prompt["split"], prompt["key"].stem)
        return

    app = EditorApp(args, prompts)
    demo = build_ui(app)
    demo.launch(server_name=args.server_name, server_port=args.server_port, css=EDITOR_CSS, footer_links=[])


class EditorApp:
    def __init__(self, args: argparse.Namespace, prompts: list[dict[str, Any]]):
        self.args = args
        self.prompts = prompts
        self.keys = [prompt["key"] for prompt in prompts]
        self.key_by_stem = {key.stem: key for key in self.keys}
        self.prompt_by_stem = {prompt["key"].stem: prompt for prompt in prompts}
        self.submitted_stems = submitted_stems(args.output_dir)
        self.predictor: SamPredictor | None = None

    def predictor_for(self, image: np.ndarray, stem: str) -> SamPredictor:
        if self.predictor is None:
            sam = sam_model_registry[self.args.sam_model](checkpoint=str(self.args.sam_checkpoint))
            sam.to(device=self.args.device)
            self.predictor = SamPredictor(sam)
            self._predictor_stem = None
        if self._predictor_stem != stem:
            self.predictor.set_image(image)
            self._predictor_stem = stem
        return self.predictor

    def new_state(self, stem: str) -> dict[str, Any]:
        key = self.key_by_stem[stem]
        frame = load_frames_by_index(key.video_id)[key.frame_index]
        height, width = annotation_shape(frame)
        rgb = cv2.resize(read_rgb(sparse_frame_path(key)), (width, height), interpolation=cv2.INTER_AREA)
        return {
            "stem": stem,
            "split": self.prompt_by_stem.get(stem, {}).get("split", "unknown"),
            "rgb": rgb,
            "points": [],
            "point_labels": [],
            "accepted": [],
            "proposal": np.zeros((height, width), dtype=bool),
        }

    def remaining_stems(self) -> list[str]:
        return [key.stem for key in self.keys if key.stem not in self.submitted_stems]

    def save(self, state: dict[str, Any]) -> None:
        stem = state["stem"]
        split = state.get("split", "unknown")
        mask = submitted_mask(state)
        frame_dir = self.args.output_dir / split / stem
        frame_dir.mkdir(parents=True, exist_ok=True)
        mask_path = frame_dir / "active_object_mask.png"
        overlay_path = frame_dir / "overlay.png"
        meta_path = frame_dir / "metadata.json"
        cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
        overlay = blend_mask(state["rgb"], mask, (255, 210, 40), 0.60)
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        metadata = {
            "frame": stem,
            "split": split,
            "label": "active_object",
            "mask": str(mask_path.relative_to(ROOT)),
            "overlay": str(overlay_path.relative_to(ROOT)),
            "shape": list(mask.shape),
            "mask_pixels": int(mask.sum()),
            "is_empty": bool(not np.any(mask)),
            "accepted_regions": len(state["accepted"]),
            "used_pending_proposal": bool(np.any(state["proposal"])),
            "points": state["points"],
            "point_labels": state["point_labels"],
        }
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        self.submitted_stems.add(stem)


def build_ui(app: EditorApp):
    stems = app.remaining_stems()
    initial = app.new_state(stems[0]) if stems else empty_state()
    with gr.Blocks(title="Active Object HITL Ground Truth Editor", fill_height=True) as demo:
        state = gr.State(initial)
        gr.Markdown("# Active Object HITL Ground Truth Editor")
        with gr.Row():
            frame = gr.Dropdown(stems, value=stems[0] if stems else None, label="Frame")
            mode = gr.Radio(["positive", "negative"], value="positive", label="Click mode")
        image = gr.Image(
            value=render(initial),
            label="Click image to add SAM points",
            type="numpy",
            interactive=True,
            sources=[],
            buttons=["fullscreen"],
            elem_id="editor-image",
        )
        with gr.Row():
            accept = gr.Button("Accept Proposal", variant="primary", elem_id="accept-proposal-button")
            undo_point = gr.Button("Undo Point", elem_id="undo-point-button")
            clear_points = gr.Button("Clear Proposal Points")
            undo_region = gr.Button("Undo Accepted Region")
            clear_frame = gr.Button("Clear Frame")
        with gr.Row():
            save = gr.Button("Submit HITL Ground Truth", variant="primary", elem_id="submit-ground-truth-button")

        demo.load(lambda: refresh_dataset(app), outputs=[frame, state, image], js=KEYBOARD_JS)
        frame.change(lambda stem: load_selected(app, stem), inputs=frame, outputs=[state, image])
        def on_image_select(st, point_mode, evt: gr.SelectData):
            return click_image(app, st, point_mode, evt)

        image.select(on_image_select, inputs=[state, mode], outputs=[state, image])
        accept.click(accept_proposal, inputs=state, outputs=[state, image])
        undo_point.click(lambda st: undo_last_point(app, st), inputs=state, outputs=[state, image])
        clear_points.click(clear_proposal_points, inputs=state, outputs=[state, image])
        undo_region.click(undo_accepted_region, inputs=state, outputs=[state, image])
        clear_frame.click(clear_current_frame, inputs=state, outputs=[state, image])
        save.click(
            lambda st: submit_and_advance(app, st),
            inputs=state,
            outputs=[frame, state, image],
        )
    return demo


def load_selected(app: EditorApp, stem: str):
    if stem is None:
        state = empty_state()
        return state, completion_image(state)
    state = app.new_state(stem)
    return state, render(state)


def refresh_dataset(app: EditorApp):
    app.submitted_stems = submitted_stems(app.args.output_dir)
    stems = app.remaining_stems()
    if not stems:
        state = empty_state()
        return gr.update(choices=[], value=None), state, completion_image(state)
    state = app.new_state(stems[0])
    return gr.update(choices=stems, value=stems[0]), state, render(state)


def click_image(app: EditorApp, state: dict[str, Any], point_mode: str, evt: gr.SelectData | None = None):
    if evt is None:
        return state, render(state)
    x, y = evt.index
    state["points"].append([float(x), float(y)])
    state["point_labels"].append(1 if point_mode == "positive" else 0)
    state["proposal"] = predict_proposal(app, state)
    return state, render(state)


def predict_proposal(app: EditorApp, state: dict[str, Any]) -> np.ndarray:
    if not state["points"]:
        return np.zeros(state["rgb"].shape[:2], dtype=bool)
    predictor = app.predictor_for(state["rgb"], state["stem"])
    masks, scores, _ = predictor.predict(
        point_coords=np.asarray(state["points"], dtype=np.float32),
        point_labels=np.asarray(state["point_labels"], dtype=np.int32),
        multimask_output=True,
    )
    return masks[int(np.argmax(scores))].astype(bool)


def accept_proposal(state: dict[str, Any]):
    if np.count_nonzero(state["proposal"]) == 0:
        return state, render(state)
    state["accepted"].append(state["proposal"].copy())
    state["points"] = []
    state["point_labels"] = []
    state["proposal"] = np.zeros(state["rgb"].shape[:2], dtype=bool)
    return state, render(state)


def undo_last_point(app: EditorApp, state: dict[str, Any]):
    if state["points"]:
        state["points"].pop()
        state["point_labels"].pop()
    state["proposal"] = predict_proposal(app, state) if state["points"] else np.zeros(state["rgb"].shape[:2], dtype=bool)
    return state, render(state)


def clear_proposal_points(state: dict[str, Any]):
    state["points"] = []
    state["point_labels"] = []
    state["proposal"] = np.zeros(state["rgb"].shape[:2], dtype=bool)
    return state, render(state)


def undo_accepted_region(state: dict[str, Any]):
    if state["accepted"]:
        state["accepted"].pop()
    return state, render(state)


def clear_current_frame(state: dict[str, Any]):
    state["points"] = []
    state["point_labels"] = []
    state["accepted"] = []
    state["proposal"] = np.zeros(state["rgb"].shape[:2], dtype=bool)
    return state, render(state)


def submit_and_advance(app: EditorApp, state: dict[str, Any]):
    app.save(state)
    stems = app.remaining_stems()
    if not stems:
        return gr.update(choices=[], value=None), state, completion_image(state)
    current_keys = [key.stem for key in app.keys]
    start_index = current_keys.index(state["stem"])
    next_stem = next((stem for stem in current_keys[start_index + 1 :] + current_keys[:start_index] if stem in stems), stems[0])
    next_state = app.new_state(next_stem)
    return gr.update(choices=stems, value=next_stem), next_state, render(next_state)


def render(state: dict[str, Any]) -> np.ndarray:
    image = state["rgb"].copy()
    accepted = union_masks(state)
    image = blend_mask(image, accepted, (255, 210, 40), 0.55)
    image = blend_mask(image, state["proposal"], (0, 220, 80), 0.50)
    draw_mask_outline(image, accepted, (255, 210, 40), thickness=2)
    draw_mask_outline(image, state["proposal"], (0, 255, 80), thickness=3)
    for point, label in zip(state["points"], state["point_labels"]):
        color = (0, 255, 0) if label == 1 else (255, 40, 40)
        cv2.circle(image, (int(point[0]), int(point[1])), 5, color, thickness=-1)
        cv2.circle(image, (int(point[0]), int(point[1])), 6, (0, 0, 0), thickness=1)
    put_header(image, f"{state.get('split', 'unknown')} | {state['stem']} | accepted={len(state['accepted'])} | points={len(state['points'])}")
    return image


def empty_state() -> dict[str, Any]:
    return {
        "stem": "",
        "split": "complete",
        "rgb": np.zeros((480, 854, 3), dtype=np.uint8),
        "points": [],
        "point_labels": [],
        "accepted": [],
        "proposal": np.zeros((480, 854), dtype=bool),
    }


def completion_image(state: dict[str, Any]) -> np.ndarray:
    image = state["rgb"].copy()
    put_header(image, "All prompted frames have been submitted.")
    return image


def union_masks(state: dict[str, Any]) -> np.ndarray:
    mask = np.zeros(state["rgb"].shape[:2], dtype=bool)
    for item in state["accepted"]:
        mask |= item.astype(bool)
    return mask


def submitted_mask(state: dict[str, Any]) -> np.ndarray:
    mask = union_masks(state)
    mask |= state["proposal"].astype(bool)
    return mask


def blend_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    out = image.astype(np.float32)
    colors = np.zeros_like(out)
    colors[mask] = color
    out[mask] = (1.0 - alpha) * out[mask] + alpha * colors[mask]
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_mask_outline(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
    if not np.any(mask):
        return
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color, thickness=thickness)


def put_header(image: np.ndarray, text: str) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 30), (0, 0, 0), thickness=-1)
    cv2.putText(image, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)


def load_frame_prompts(manifest_path: Path, split: str) -> list[dict[str, Any]]:
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        frames = data.get("frames") or data.get("samples") or []
        prompts = [
            {
                "key": FrameKey.from_stem(item["frame"]),
                "split": item.get("split", split),
                "source_split": item.get("source_split", item.get("split", split)),
            }
            for item in frames
        ]
        if prompts:
            return prompts
    return [{"key": key, "split": split, "source_split": split} for key in split_frame_keys(split)]


def submitted_stems(output_dir: Path) -> set[str]:
    stems = set()
    for split_dir in output_dir.iterdir() if output_dir.exists() else []:
        if not split_dir.is_dir():
            continue
        for frame_dir in split_dir.iterdir():
            if frame_dir.is_dir():
                stems.add(frame_dir.name)
    return stems


if __name__ == "__main__":
    main()
