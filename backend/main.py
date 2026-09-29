import io
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import requests
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

# --- Projekt gyökérútvonalának meghatározása ---
CURRENT_FILE = Path(__file__).resolve()
BASE_DIR = CURRENT_FILE.parent.parent if CURRENT_FILE.parent.name == "backend" else CURRENT_FILE.parent

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# --- FastAPI App Setup ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Használt eszköz: {device}")

# --- Tokenizer & Kép transzformációk ---
print("⏳ BERT Tokenizer és Kép transzformációk betöltése...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- MODELL STRUKTÚRÁK DEFINIÁLÁSA ---

# 1. KÉPI MODELL (ResNet18 - a Kaggle tanításod alapján)
def get_image_model():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, 2)
    )
    return model

# 2. SZÖVEGES MODELL (BERT)
class TextClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        feat = getattr(out, 'pooler_output', out.last_hidden_state[:, 0, :])
        return self.classifier(feat)


# --- 2 KÜLÖN MODELL BETÖLTÉSE ---
print("⏳ Modellfájlok betöltése...")

MODEL_DIR = BASE_DIR / "backend" / "model"
if not MODEL_DIR.exists():
    MODEL_DIR = BASE_DIR / "model"

text_model_path = MODEL_DIR / "text_only_bert.pt"
image_model_path = MODEL_DIR / "best_image_model.pt"

# Útvonalak ellenőrzése
if not text_model_path.exists():
    raise FileNotFoundError(f"❌ A szöveges modell nem található itt: {text_model_path}")
if not image_model_path.exists():
    raise FileNotFoundError(f"❌ A képi modell nem található itt: {image_model_path}")

# Szöveges modell betöltése
text_model = TextClassifier().to(device)
try:
    text_model.load_state_dict(torch.load(text_model_path, map_location=device))
except Exception:
    # Ha AutoModelForSequenceClassification-nel lett elmentve:
    text_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)
    text_model.load_state_dict(torch.load(text_model_path, map_location=device))

text_model.eval()
print(f"✅ Szöveges modell betöltve: {text_model_path.name}")

# Képi modell betöltése
image_model = get_image_model().to(device)
image_model.load_state_dict(torch.load(image_model_path, map_location=device))
image_model.eval()
print(f"✅ Képi modell betöltve: {image_model_path.name}")


# --- Útvonalak (Routes) ---
@app.get("/")
def health_check():
    return {"status": "ok", "message": "Fake News Detection API is running"}


@app.get("/predict")
def predict_info():
    return {
        "status": "info",
        "message": "A /predict végpont POST kérést vár (szöveg, képfájl vagy képlink).",
        "docs": "http://localhost:8000/docs",
        "frontend": "http://localhost:5173"
    }


@app.post("/predict")
async def predict(
    text: str = Form(None),
    image_url: str = Form(None),
    file: UploadFile = File(None)
):
    image = None
    has_text = bool(text and text.strip() != "")

    # 1. Kép beolvasása (Fájlból vagy URL-ből)
    if file is not None and file.filename != "":
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            return {"error": f"Érvénytelen képfájl: {e}"}
            
    elif image_url and image_url.strip().lower() not in ["", "string", "null", "undefined"]:
        clean_url = image_url.strip()
        if not (clean_url.startswith("http://") or clean_url.startswith("https://")):
            return {"error": f"Érvénytelen URL formátum: '{clean_url}'."}
            
        try:
            resp = requests.get(clean_url, timeout=5)
            resp.raise_for_status()
            image = Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception as e:
            return {"error": f"Nem sikerült betölteni a képet az URL-ről: {e}"}

    has_image = bool(image is not None)

    if not has_text and not has_image:
        return {"error": "Adj meg legalább egy szöveget vagy egy képet!"}

    # 2. Előrejelzés kiszámítása
    text_probs = None
    image_probs = None

    with torch.no_grad():
        # Csak szöveg futtatása (ha van szöveg)
        if has_text:
            encoding = tokenizer(
                text.strip(),
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            ).to(device)
            text_out = text_model(input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask'])
            if hasattr(text_out, 'logits'):
                text_out = text_out.logits
            text_probs = torch.softmax(text_out, dim=1)[0]

        # Csak kép futtatása (ha van kép)
        if has_image:
            img_tensor = image_transform(image).unsqueeze(0).to(device)
            image_out = image_model(img_tensor)
            image_probs = torch.softmax(image_out, dim=1)[0]

        # 3. Kimenetek kombinálása (Ensemble / Late Fusion)
        if has_text and has_image:
            # Ha mindkettő megvan -> 50%-50% súlyozott átlag
            final_probs = 0.5 * text_probs + 0.5 * image_probs
            mode = "multimodal"
        elif has_text:
            final_probs = text_probs
            mode = "text_only"
        else:
            final_probs = image_probs
            mode = "image_only"

    real_prob = round(float(final_probs[0]) * 100, 2)
    fake_prob = round(float(final_probs[1]) * 100, 2)

    return {
        "real_prob": real_prob,
        "fake_prob": fake_prob,
        "label": 0 if real_prob > fake_prob else 1,
        "has_text": has_text,
        "has_image": has_image,
        "mode": mode
    }