from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import io
import requests
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import timm
from torchvision import transforms
import torch.nn.functional as F

# --- App setup ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

# --- Frozen BERT (text encoder) ---
print("⏳ Loading BERT...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_encoder = AutoModel.from_pretrained("bert-base-uncased").to(device).eval()
for p in bert_encoder.parameters():
    p.requires_grad = False
print("✅ BERT loaded and frozen")

# --- Frozen ViT (image encoder) ---
print("⏳ Loading ViT...")
vit_encoder = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0).to(device).eval()
for p in vit_encoder.parameters():
    p.requires_grad = False
print("✅ ViT loaded and frozen")

vit_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Classifier head (the only trained component) ---
class MultimodalFusionClassifier(nn.Module):
    def __init__(self, fusion_dim=1536, hidden_dim=256, num_classes=2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, text_feat, image_feat):
        combined = torch.cat((text_feat, image_feat), dim=1)
        return self.classifier(combined)

print("⏳ Loading classifier head...")
model = MultimodalFusionClassifier().to(device)
model.load_state_dict(torch.load("model/classifier_head.pt", map_location=device))
model.eval()
print("✅ Classifier head loaded — model ready!")


# --- Helper functions ---

def get_text_embedding(text: str):
    encoding = tokenizer(
        text, truncation=True, padding='max_length', max_length=128, return_tensors='pt'
    ).to(device)
    with torch.no_grad():
        return bert_encoder(**encoding).last_hidden_state[:, 0, :]  # no normalization


def get_image_embedding(image=None):
    if image is not None:
        img_tensor = vit_transform(image.convert("RGB")).unsqueeze(0).to(device)
    else:
        img_tensor = torch.zeros(1, 3, 224, 224).to(device)  # blank image, matches training fallback
    with torch.no_grad():
        return vit_encoder(img_tensor)  # no normalization

# --- Routes ---
@app.get("/")
def health_check():
    return {"status": "ok", "message": "Fake News Detection API is running"}


@app.post("/predict")
async def predict(
    text: str = Form(...),
    image_url: str = Form(None),
    file: UploadFile = File(None)
):
    image = None

    if file is not None:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    elif image_url:
        try:
            resp = requests.get(image_url, timeout=5)
            image = Image.open(io.BytesIO(resp.content))
        except Exception as e:
            return {"error": f"Could not load image from URL: {e}"}

    text_feat = get_text_embedding(text)
    image_feat = get_image_embedding(image)

    with torch.no_grad():
        outputs = model(text_feat, image_feat)
        print("Raw Logits:", outputs.cpu().numpy())
        probs = torch.softmax(outputs, dim=1)[0]
        print("Softmax Probs:", probs.cpu().numpy())

    return {
        "real_prob": round(float(probs[0]) * 100, 2),
        "fake_prob": round(float(probs[1]) * 100, 2),
        "has_image": image is not None
    }