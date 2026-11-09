import io
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import requests
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# --- Útvonalak: a projekt gyökere (az src csomag miatt) és a backend mappa (legacy.py) ---
BACKEND_DIR = Path(__file__).resolve().parent
BASE_DIR = BACKEND_DIR.parent
for p in (str(BASE_DIR), str(BACKEND_DIR)):
    if p not in sys.path:
        sys.path.append(p)

from src.explain.shap_analysis import contributions_for_api, load_head, modality_shapley  # noqa: E402
from src.models.encoders import (MODALITY_NAMES, CaptionGenerator, ClipScorer,  # noqa: E402
                                 ImageEncoder, TextEncoder)

MODEL_DIR = BACKEND_DIR / "model"
FUSION_CKPT = Path(os.environ.get("FUSION_CKPT", MODEL_DIR / "fusion_head.pt"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FusionPipeline:
    """Fagyasztott enkóderek (ugyanaz a kód, mint a tanításnál) + betanított fúziós fej."""
    engine = "fusion"

    def __init__(self, ckpt_path):
        self.head, self.ck = load_head(ckpt_path, device)
        self.features = self.ck["features"]
        self.names = [MODALITY_NAMES[f] for f in self.features]
        needs_text = any(f in ("text_bert", "caption_bert") for f in self.features)
        self.text_enc = TextEncoder(device=device) if needs_text else None
        self.img_enc = None
        if "img_vit" in self.features:
            self.img_enc = ImageEncoder("vit", device=device)
        elif "img_resnet18" in self.features:
            self.img_enc = ImageEncoder("resnet18", device=device)
        self.captioner = CaptionGenerator(device=device) if "caption_bert" in self.features else None
        self.clip = ClipScorer(device=device) if "clip_score" in self.features else None

    def info(self):
        t = self.ck.get("test", {})
        return {"engine": self.engine, "model": self.ck.get("name"), "seed": self.ck.get("seed"),
                "features": self.features, "fusion": "concat + MLP",
                "test_f1": t.get("f1"), "test_acc": t.get("acc"), "test_auc": t.get("auc")}

    @torch.no_grad()
    def predict(self, text, image):
        caption = self.captioner.generate([image])[0] if (self.captioner and image is not None) else None
        feats, masks = [], []
        for f in self.features:
            if f == "text_bert":
                ok, vec = bool(text), self.text_enc.encode([text or ""])
            elif f == "caption_bert":
                ok, vec = bool(caption), self.text_enc.encode([caption or "[no image]"])
            elif f == "clip_score":
                ok = image is not None and bool(text)
                vec = self.clip.score([text or ""], [image if ok else None])
            else:
                ok = image is not None
                vec = self.img_enc.encode([image]) if ok else torch.zeros(1, self.head.dims[len(feats)]).numpy()
            feats.append(torch.from_numpy(vec).float().to(device))
            masks.append(torch.tensor([ok], device=device))
        phi, _, prob = modality_shapley(self.head, feats, masks)
        present = [i for i, m in enumerate(masks) if bool(m[0])]
        contrib = contributions_for_api(phi[0, present].cpu().numpy(), [self.names[i] for i in present])
        return {"prob_fake": float(prob[0]), "contributions": contrib, "caption": caption}


print(f"🖥️ Használt eszköz: {device}")
if FUSION_CKPT.exists():
    print(f"⏳ Fúziós modell betöltése: {FUSION_CKPT.name}")
    pipeline = FusionPipeline(FUSION_CKPT)
else:
    from legacy import LegacyPipeline
    print(f"⚠️ Nincs {FUSION_CKPT.name}, a régi (WELFake + ResNet18) modellek töltődnek be.")
    pipeline = LegacyPipeline(MODEL_DIR, device)
print(f"✅ Kész: {pipeline.info()}")

app = FastAPI(title="Fake News Detection API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Fake News Detection API is running"}


@app.get("/model-info")
def model_info():
    return pipeline.info()


@app.get("/predict")
def predict_info():
    return {"status": "info", "message": "A /predict végpont POST kérést vár (szöveg, képfájl vagy képlink).",
            "docs": "http://localhost:8000/docs", "frontend": "http://localhost:5173"}


@app.post("/predict")
async def predict(text: str = Form(None), image_url: str = Form(None), file: UploadFile = File(None)):
    image = None
    text = (text or "").strip()
    has_text = text != ""

    if file is not None and file.filename:
        try:
            image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        except Exception as e:
            return {"error": f"Érvénytelen képfájl: {e}"}
    elif image_url and image_url.strip().lower() not in ["", "string", "null", "undefined"]:
        url = image_url.strip()
        if not url.startswith(("http://", "https://")):
            return {"error": f"Érvénytelen URL formátum: '{url}'."}
        try:
            resp = requests.get(url, timeout=5, headers={"User-Agent": "FakeNewsDetection/1.0"})
            resp.raise_for_status()
            image = Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception as e:
            return {"error": f"Nem sikerült betölteni a képet az URL-ről: {e}"}

    has_image = image is not None
    if not has_text and not has_image:
        return {"error": "Adj meg legalább egy szöveget vagy egy képet!"}

    out = pipeline.predict(text if has_text else None, image)
    fake_prob = round(out["prob_fake"] * 100, 2)
    real_prob = round(100 - fake_prob, 2)
    mode = "multimodal" if has_text and has_image else ("text_only" if has_text else "image_only")
    return {
        "real_prob": real_prob,
        "fake_prob": fake_prob,
        "label": 0 if real_prob > fake_prob else 1,
        "has_text": has_text,
        "has_image": has_image,
        "mode": mode,
        "engine": pipeline.engine,
        "contributions": out["contributions"],
        "caption": out["caption"],
    }
