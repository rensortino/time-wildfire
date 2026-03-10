#!/usr/bin/env python3
"""
Gradio interface for SAM3 bounding box tracking.

Upload a sequence of images, draw a bounding box on the first frame,
and track the object across all frames using SAM3.
"""

import gradio as gr
from PIL import Image

from sam3_tracker import SAM3Tracker, draw_bbox, draw_legend


def _load_images_from_gradio(input_images: list) -> list[Image.Image]:
    images = []
    for item in input_images:
        path = item if isinstance(item, str) else item.name if hasattr(item, "name") else str(item)
        images.append(Image.open(path).convert("RGB"))
    return images


def run_tracking(input_images: list, bbox_x_min: float, bbox_y_min: float,
                 bbox_x_max: float, bbox_y_max: float):
    """Gradio callback: validate inputs, run tracking, return gallery."""
    if not input_images or len(input_images) < 2:
        raise gr.Error("Please upload at least 2 images.")

    images = _load_images_from_gradio(input_images)

    w, h = images[0].size
    bbox = [
        max(0, min(bbox_x_min, w)),
        max(0, min(bbox_y_min, h)),
        max(0, min(bbox_x_max, w)),
        max(0, min(bbox_y_max, h)),
    ]

    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise gr.Error(
            f"Invalid bounding box: x_max must be > x_min and y_max must be > y_min. "
            f"Got [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]."
        )

    tracker = SAM3Tracker()
    predicted_bboxes = tracker.track(images, bbox)

    legend_entries = [("Input", "cyan"), ("Tracked", "yellow")]
    result_images = []
    for idx, img in enumerate(images):
        annotated = img.copy()
        if idx in predicted_bboxes:
            color = "yellow" if idx > 0 else "cyan"
            annotated = draw_bbox(annotated, predicted_bboxes[idx], color=color)
        annotated = draw_legend(annotated, legend_entries)
        result_images.append(annotated)

    return result_images


def preview_bbox(input_images: list, bbox_x_min: float, bbox_y_min: float,
                 bbox_x_max: float, bbox_y_max: float):
    """Show the bounding box drawn on the first frame for visual confirmation."""
    if not input_images:
        return None

    images = _load_images_from_gradio(input_images[:1])
    img = images[0]
    w, h = img.size
    bbox = [
        max(0, min(bbox_x_min, w)),
        max(0, min(bbox_y_min, h)),
        max(0, min(bbox_x_max, w)),
        max(0, min(bbox_y_max, h)),
    ]
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return img
    return draw_bbox(img, bbox, color="cyan", label="Input BBox")


def build_app():
    with gr.Blocks(title="SAM3 Bounding Box Tracker", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# SAM3 Bounding Box Tracker\n"
            "Upload a sequence of images (video frames), specify a bounding box on the first frame, "
            "and track the object across the entire sequence using **SAM3**."
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_gallery = gr.File(
                    label="Upload image sequence (in order)",
                    file_count="multiple",
                    file_types=["image"],
                )
                gr.Markdown("### Bounding box on first frame (pixels)")
                with gr.Row():
                    bbox_x_min = gr.Number(label="x_min", value=0, precision=0)
                    bbox_y_min = gr.Number(label="y_min", value=0, precision=0)
                with gr.Row():
                    bbox_x_max = gr.Number(label="x_max", value=100, precision=0)
                    bbox_y_max = gr.Number(label="y_max", value=100, precision=0)

                preview_btn = gr.Button("Preview bounding box", variant="secondary")
                preview_image = gr.Image(label="First frame with bounding box", type="pil")

                run_btn = gr.Button("Run tracking", variant="primary")

            with gr.Column(scale=2):
                output_gallery = gr.Gallery(
                    label="Tracking results",
                    columns=4,
                    object_fit="contain",
                    height="auto",
                )

        bbox_inputs = [input_gallery, bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max]

        preview_btn.click(fn=preview_bbox, inputs=bbox_inputs, outputs=preview_image)
        run_btn.click(fn=run_tracking, inputs=bbox_inputs, outputs=output_gallery)

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch()
