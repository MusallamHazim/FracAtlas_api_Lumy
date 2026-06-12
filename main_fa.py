from fastapi import FastAPI, UploadFile, File, HTTPException
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b3
from PIL import Image
import io

# 1. Initialize FastAPI app
app = FastAPI(title="Luminary FracAtlas API")
device = torch.device("cpu")

# ==========================================
# 2. REBUILD THE EFFICIENTNET-B3 ARCHITECTURE
# ==========================================
model = efficientnet_b3()
# Override the final classifier layer to match the 2 classes
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 2)

# ==========================================
# 3. UNPACK THE CHECKPOINT SUITCASE
# ==========================================
MODEL_PATH = "efficientnetb3_fracatlas_final_checkpoint.pth.zip"

try:
    # weights_only=False because this is a dictionary containing strings/lists
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("FracAtlas model loaded successfully.")
except Exception as e:
    print(f"Failed to load model. Error: {e}")

# ==========================================
# 4. THE CORRECT 300x300 TRANSFORMS
# ==========================================
inference_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# From your dataset's id_to_label mapping
class_names = ["not_fractured", "fractured"]

# ==========================================
# 5. THE PREDICTION ENDPOINT
# ==========================================
@app.post("/analyze-fracture")
async def predict_fracture(file: UploadFile = File(...)):
    # Guardrail: Ensure file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Provided file is not an image.")
        
    try:
        # Read the image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Apply the 300x300 transforms and add batch dimension
        tensor = inference_transform(image).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            prob_not_fractured = probabilities[0][0].item()
            prob_fractured = probabilities[0][1].item()
            
        # =========================================================
        # MEDICAL SENSITIVITY THRESHOLD
        # Because the dataset is heavily biased towards healthy bones,
        # we lower the threshold to catch more potential fractures.
        # =========================================================
        SENSITIVITY_THRESHOLD = 0.30 
        
        if prob_fractured >= SENSITIVITY_THRESHOLD:
            pred_class = "fractured"
            final_confidence = prob_fractured
        else:
            pred_class = "not_fractured"
            final_confidence = prob_not_fractured
            
        conf_score = round(final_confidence * 100, 2)
        
        return {
            "filename": file.filename,
            "prediction": pred_class,
            "confidence": f"{conf_score}%",
            "probabilities": {
                "not_fractured": round(prob_not_fractured, 4),
                "fractured": round(prob_fractured, 4)
            },
            "threshold_used": SENSITIVITY_THRESHOLD
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "healthy", "message": "FracAtlas API is running!"}