#!/usr/bin/env python3
"""
Core SAM3 tracking logic shared by the Gradio app and the CLI script.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Optional

from transformers import Sam3TrackerVideoProcessor, Sam3TrackerVideoModel
import torch

from utils import mask_to_bbox


class SAM3Tracker:
    """Lazy-loading wrapper around the SAM3 video tracking model."""

    def __init__(self, model_name: str = "facebook/sam3", device: Optional[str] = None):
        self._model_name = model_name
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor: Optional[Sam3TrackerVideoProcessor] = None
        self._model: Optional[Sam3TrackerVideoModel] = None

    def _load(self):
        if self._processor is None:
            self._processor = Sam3TrackerVideoProcessor.from_pretrained(self._model_name)
            self._model = Sam3TrackerVideoModel.from_pretrained(self._model_name)
            self._model = self._model.to(self._device)
            self._model.eval()

    def track(
        self,
        images: list[Image.Image],
        bbox: list[float],
    ) -> dict[int, list[float]]:
        """
        Run SAM3 tracking on a sequence of PIL images.

        Args:
            images: ordered list of PIL RGB images (video frames).
            bbox: first-frame bounding box [x_min, y_min, x_max, y_max] in pixels.

        Returns:
            dict mapping frame index -> predicted bbox [x_min, y_min, x_max, y_max].
            Frame 0 always maps to the input bbox.
        """
        self._load()

        inference_session = self._processor.init_video_session(
            video=images,
            inference_device=self._device,
            dtype=torch.float32,
        )

        input_boxes = [[[float(c) for c in bbox]]]
        self._processor.add_inputs_to_inference_session(
            inference_session=inference_session,
            frame_idx=0,
            obj_ids=[1],
            input_boxes=input_boxes,
        )

        predicted: dict[int, list[float]] = {0: list(bbox)}

        for output in self._model.propagate_in_video_iterator(
            inference_session, start_frame_idx=0
        ):
            if output.frame_idx == 0:
                continue

            masks = self._processor.post_process_masks(
                [output.pred_masks],
                original_sizes=[
                    [inference_session.video_height, inference_session.video_width]
                ],
                binarize=True,
            )[0]

            if len(masks) > 0:
                bb = mask_to_bbox(masks[0].squeeze())
                if bb.size > 0:
                    predicted[output.frame_idx] = bb.tolist()

        return predicted


def draw_bbox(
    image: Image.Image,
    bbox: list[float],
    color: str = "lime",
    width: int = 3,
) -> Image.Image:
    """Draw a bounding box on a copy of *image* (no text label)."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    x_min, y_min, x_max, y_max = bbox
    draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=width)
    return img


def draw_legend(
    image: Image.Image,
    entries: list[tuple[str, str]],
    margin: int = 10,
    swatch_size: int = 16,
    spacing: int = 6,
) -> Image.Image:
    """
    Draw a colour legend in the top-right corner of *image*.

    *entries* is a list of ``(label, colour)`` pairs, e.g.
    ``[("GT", "blue"), ("Pred", "yellow")]``.
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14
        )
    except Exception:
        font = ImageFont.load_default()

    line_height = max(swatch_size, font.size if hasattr(font, "size") else 14)
    row_height = line_height + spacing

    text_widths = [draw.textlength(label, font=font) for label, _ in entries]
    box_w = int(swatch_size + spacing + max(text_widths) + 2 * margin)
    box_h = int(len(entries) * row_height - spacing + 2 * margin)

    x0 = img.width - box_w - margin
    y0 = margin

    draw.rectangle([x0, y0, x0 + box_w, y0 + box_h], fill=(0, 0, 0, 180))

    for i, (label, color) in enumerate(entries):
        row_y = y0 + margin + i * row_height
        sw_y = row_y + (line_height - swatch_size) // 2
        draw.rectangle(
            [x0 + margin, sw_y, x0 + margin + swatch_size, sw_y + swatch_size],
            fill=color,
            outline=color,
        )
        draw.text(
            (x0 + margin + swatch_size + spacing, row_y),
            label,
            fill="white",
            font=font,
        )

    return img
