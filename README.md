# PPE Detection Web Application

This project is a web-based Personal Protective Equipment (PPE) detection system built using YOLOv8 and FastAPI.

## Getting Started
1) Place your trained weights at `backend/model/best.pt`.
2) Install dependencies: `pip install -r backend/requirements.txt`
3) Run the API: `uvicorn backend.main:app --reload`
4) Open the UI: visit `http://127.0.0.1:8000` (served by FastAPI) and upload an image.

## Detected Classes
- Mask
- Safety Helmet
- Safety Vest

## Model
- YOLOv8
- Achieved **93.7% mAP@0.5**

## Tech Stack
- Python
- FastAPI
- YOLOv8 (Ultralytics)
- HTML / JavaScript

## Note
Model weights are not included in the repository.
