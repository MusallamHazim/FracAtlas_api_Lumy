# Luminary FracAtlas API

EfficientNet-B3 model trained on the FracAtlas dataset to classify X-ray images as fractured / not fractured. Part of the Luminary (Lumy) graduation project.

## Files
- `main_fa.py` — FastAPI app
- `efficientnetb3_fracatlas_final_checkpoint.pth.zip` — trained model checkpoint

## Run locally
```bash
pip install -r requirements.txt
uvicorn main_fa:app --reload --port 8000
```
Then open `http://127.0.0.1:8000/docs` to test.

## Endpoint
- `POST /analyze-fracture` — upload an X-ray image, returns prediction + confidence.
