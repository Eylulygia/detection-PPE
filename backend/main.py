import os
os.environ["TORCH_WEIGHTS_ONLY"] = "0"

import base64
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from ultralytics import YOLO
import torch

app = FastAPI() 

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_INDEX = BASE_DIR.parent / "frontend" / "index.html"
MODEL_PATH = BASE_DIR / "model" / "best.pt"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model weights not found at {MODEL_PATH}. "
        "Place your trained best.pt there before starting the server."
    )

# PyTorch 2.6+ defaults to weights_only=True; force False for trusted local checkpoints.
_torch_load = torch.load


def _torch_load_with_weights_only_false(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _torch_load_with_weights_only_false  # type: ignore

model = YOLO(str(MODEL_PATH))
CLASS_COLORS = {
    "Mask": (255, 87, 34),
    "Safety Helmet": (46, 204, 113),
    "Safety-Helmet": (46, 204, 113),
    "Safety Vest": (52, 152, 219),
    "Safety-Vest": (52, 152, 219),
}


def _label_for_class(cls_id: int) -> str:
    """Return readable class label from the YOLO model names."""
    if isinstance(model.names, dict):
        return model.names.get(cls_id, str(cls_id))
    return str(model.names[cls_id])


def _draw_label(canvas: np.ndarray, text: str, x1: float, y1: float, color: tuple[int, int, int]) -> None:
    """Draw a solid label background and text above the bounding box."""
    label_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    label_w, label_h = label_size
    top_left_y = max(int(y1) - label_h - 6, 0)
    cv2.rectangle(
        canvas,
        (int(x1), top_left_y),
        (int(x1) + label_w + 6, int(y1)),
        color,
        thickness=-1,
    )
    cv2.putText(
        canvas,
        text,
        (int(x1) + 3, int(y1) - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        thickness=1,
        lineType=cv2.LINE_AA,
    )


@app.get("/")
async def serve_frontend() -> FileResponse:
    if FRONTEND_INDEX.exists():
        return FileResponse(FRONTEND_INDEX)
    raise HTTPException(status_code=404, detail="Frontend not found")


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="No file uploaded")

    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")

    results = model.predict(img, imgsz=640, conf=0.25, verbose=False)
    result = results[0]

    detections = []
    annotated = img.copy()

    for box in result.boxes:
        x1, y1, x2, y2 = [float(x) for x in box.xyxy[0]]
        label = _label_for_class(int(box.cls))
        conf = float(box.conf)
        color = CLASS_COLORS.get(label, (0, 132, 255))

        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=2)
        _draw_label(annotated, f"{label} {conf:.2f}", x1, y1, color)

        detections.append(
            {
                "class": label,
                "confidence": round(conf, 4),
                "bbox": [x1, y1, x2, y2],
            }
        )

    ok, buffer = cv2.imencode(".jpg", annotated)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode annotated image")

    encoded_image = base64.b64encode(buffer.tobytes()).decode("utf-8")

    return {"detections": detections, "image": encoded_image}
