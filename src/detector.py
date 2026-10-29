"""₹50 banknote detector using OpenCV DNN on the exported YOLOv8 ONNX model."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .onnx_utils import Detection, decode_yolov8, make_blob, read_model_info

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "model.onnx"
DISPLAY_LABEL = "₹50"
FALLBACK_LABEL = "Rs.50"  # used if no font with the ₹ glyph is available

BOX_COLOR_BGR = (0, 200, 0)
TEXT_COLOR_BGR = (0, 0, 0)

# Windows fonts that include U+20B9 (₹), plus common Linux/macOS fallbacks.
_FONT_CANDIDATES = [
    "C:/Windows/Fonts/Nirmala.ttc",
    "C:/Windows/Fonts/NirmalaUI.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
]


class DetectorError(RuntimeError):
    pass


class RupeeDetector:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH,
                 conf_threshold: float = 0.6, iou_threshold: float = 0.45,
                 use_cuda: bool = False):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise DetectorError(
                f"ONNX model not found: {self.model_path}\n"
                f"Copy your exported best.onnx to {DEFAULT_MODEL_PATH}")
        if not 0.0 < conf_threshold < 1.0:
            raise DetectorError(f"conf_threshold must be in (0, 1), got {conf_threshold}")
        if not 0.0 < iou_threshold < 1.0:
            raise DetectorError(f"iou_threshold must be in (0, 1), got {iou_threshold}")

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        try:
            self.net = cv2.dnn.readNetFromONNX(str(self.model_path))
        except cv2.error as exc:
            raise DetectorError(f"OpenCV could not load {self.model_path}: {exc}") from exc

        if use_cuda:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        self.info = read_model_info(self.model_path)
        self.class_names = self.info.names or {0: "rupee_50"}
        self.num_classes = len(self.class_names)
        self.input_hw = self._resolve_input_size()
        self.output_shape = self._probe_output_shape()

        self._font = _load_font()

    def _resolve_input_size(self) -> tuple[int, int]:
        shape = self.info.input_shape
        if shape is None:
            return self._guess_input_size()  # onnx package unavailable
        if len(shape) != 4 or shape[1] != 3:
            raise DetectorError(f"Expected NCHW input with 3 channels, model has {shape}")
        h, w = shape[2], shape[3]
        if h <= 0 or w <= 0:
            raise DetectorError(f"Model has dynamic input size {shape}; re-export with a fixed imgsz")
        if h % 32 or w % 32:
            raise DetectorError(f"Input size {h}x{w} is not a multiple of the YOLOv8 stride (32)")
        return h, w

    def _guess_input_size(self) -> tuple[int, int]:
        """Without the onnx package the input shape is unknown; try common square export sizes."""
        for size in (640, 512, 416, 320, 768, 800, 1024):
            try:
                self.net.setInput(np.zeros((1, 3, size, size), np.float32))
                self.net.forward()
                return size, size
            except cv2.error:
                continue
        raise DetectorError("Could not determine the model input size; `pip install onnx` and retry")

    def _probe_output_shape(self) -> tuple[int, ...]:
        dummy = np.zeros((*self.input_hw, 3), dtype=np.uint8)
        blob, _ = make_blob(dummy, self.input_hw)
        try:
            self.net.setInput(blob)
            out = self.net.forward()
        except cv2.error as exc:
            raise DetectorError(
                f"Dry-run inference failed for input {self.input_hw}: {exc}") from exc

        expected = 4 + self.num_classes
        if out.ndim != 3 or expected not in out.shape[1:]:
            raise DetectorError(
                f"Unexpected output shape {out.shape}; expected [1, {expected}, N] or [1, N, {expected}] "
                f"for a YOLOv8 detect head with {self.num_classes} class(es)")
        return tuple(out.shape)

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise DetectorError("detect() expects a BGR image of shape (H, W, 3)")

        blob, lb = make_blob(frame_bgr, self.input_hw)
        try:
            self.net.setInput(blob)
            output = self.net.forward()
        except cv2.error as exc:
            raise DetectorError(f"Inference failed: {exc}") from exc

        return decode_yolov8(output, self.num_classes, self.conf_threshold,
                             self.iou_threshold, lb, frame_bgr.shape)

    def draw(self, frame_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Draw boxes labelled '₹50 0.93' onto a copy of the frame."""
        out = frame_bgr.copy()
        thickness = max(2, round(min(out.shape[:2]) / 300))
        for d in detections:
            cv2.rectangle(out, (d.x1, d.y1), (d.x2, d.y2), BOX_COLOR_BGR, thickness)
        if not detections:
            return out

        label_base = DISPLAY_LABEL if self._font is not None else FALLBACK_LABEL
        labels = [(d, f"{label_base} {d.confidence:.2f}") for d in detections]
        if self._font is None:
            for d, text in labels:
                _draw_label_cv2(out, text, d.x1, d.y1)
            return out
        return _draw_labels_pil(out, labels, self._font)


def _load_font():
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    for path in _FONT_CANDIDATES:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, 22)
            except OSError:
                continue
    return None


def _draw_label_cv2(img: np.ndarray, text: str, x: int, y: int) -> None:
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    top = y - th - base - 4 if y - th - base - 4 >= 0 else y
    cv2.rectangle(img, (x, top), (x + tw + 6, top + th + base + 4), BOX_COLOR_BGR, -1)
    cv2.putText(img, text, (x + 3, top + th + 2), font, scale, TEXT_COLOR_BGR, thick, cv2.LINE_AA)


def _draw_labels_pil(img_bgr: np.ndarray, labels, font) -> np.ndarray:
    from PIL import Image, ImageDraw

    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    fill = tuple(reversed(BOX_COLOR_BGR))
    text_fill = tuple(reversed(TEXT_COLOR_BGR))
    for d, text in labels:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        tw, th = right - left, bottom - top
        box_top = d.y1 - th - 8 if d.y1 - th - 8 >= 0 else d.y1
        draw.rectangle([d.x1, box_top, d.x1 + tw + 8, box_top + th + 8], fill=fill)
        draw.text((d.x1 + 4 - left, box_top + 4 - top), text, font=font, fill=text_fill)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
