"""Fagyasztott előtanított enkóderek. A tanítás és a backend is ezt importálja,
így az előfeldolgozás garantáltan ugyanaz."""
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

TEXT_MODEL = "bert-base-uncased"
DIMS = {"text_bert": 768, "img_vit": 768, "img_resnet18": 512, "caption_bert": 768, "clip_score": 1}


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextEncoder:
    """BERT [CLS] vektor (last_hidden_state[:, 0]), 768 dim."""

    def __init__(self, name=TEXT_MODEL, max_len=128, device=None):
        from transformers import AutoModel, AutoTokenizer
        self.device = device or get_device()
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(self.device).eval()
        self.max_len = max_len

    @torch.no_grad()
    def encode(self, texts):
        enc = self.tok(list(texts), truncation=True, padding=True, max_length=self.max_len,
                       return_tensors="pt").to(self.device)
        out = self.model(**enc).last_hidden_state[:, 0]
        return out.float().cpu().numpy()


class ImageEncoder:
    """kind='vit': timm vit_base_patch16_224, fej nélkül (768 dim).
    kind='resnet18': torchvision ResNet18, fc = Identity (512 dim) - CNN baseline."""

    def __init__(self, kind="vit", device=None):
        self.device = device or get_device()
        self.kind = kind
        if kind == "vit":
            import timm
            from timm.data import create_transform, resolve_data_config
            self.model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
            self.transform = create_transform(**resolve_data_config({}, model=self.model))
        elif kind == "resnet18":
            from torchvision import models
            w = models.ResNet18_Weights.DEFAULT
            self.model = models.resnet18(weights=w)
            self.model.fc = nn.Identity()
            self.transform = w.transforms()
        else:
            raise ValueError(kind)
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def encode_tensors(self, batch):
        return self.model(batch.to(self.device)).float().cpu().numpy()

    def encode(self, images):
        return self.encode_tensors(torch.stack([self.transform(im.convert("RGB")) for im in images]))


def load_image(path):
    """Biztonságos betöltés: hibás kép esetén None (-> hiányzó modalitás)."""
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def embed_dims(features):
    return [DIMS[f] for f in features]


MODALITY_NAMES = {"text_bert": "szöveg", "img_vit": "kép", "img_resnet18": "kép", "caption_bert": "képaláírás",
                  "clip_score": "szöveg–kép egyezés"}
CAPTION_MODEL = "Salesforce/blip-image-captioning-base"


class CaptionGenerator:
    """BLIP-képaláírás (3. modalitás, RQ2). A tanítás és a backend ugyanezt használja."""

    def __init__(self, name=CAPTION_MODEL, device=None, max_new_tokens=30):
        from transformers import BlipForConditionalGeneration, BlipProcessor
        self.device = device or get_device()
        self.proc = BlipProcessor.from_pretrained(name)
        self.model = BlipForConditionalGeneration.from_pretrained(name).to(self.device).eval()
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def generate(self, images):
        """images: PIL képek listája, None elemekkel; None helyén üres string."""
        valid = [im for im in images if im is not None]
        caps = []
        if valid:
            inp = self.proc(images=valid, return_tensors="pt").to(self.device)
            caps = self.proc.batch_decode(self.model.generate(**inp, max_new_tokens=self.max_new_tokens),
                                          skip_special_tokens=True)
        it = iter(caps)
        return [next(it).strip() if im is not None else "" for im in images]


CLIP_MODEL = "openai/clip-vit-base-patch32"


class ClipScorer:
    """Szöveg–kép konzisztencia (M8): a cím és a kép CLIP-embeddingjének koszinusz-hasonlósága, 1 dim."""

    def __init__(self, name=CLIP_MODEL, device=None):
        from transformers import CLIPModel, CLIPProcessor
        self.device = device or get_device()
        self.proc = CLIPProcessor.from_pretrained(name)
        self.model = CLIPModel.from_pretrained(name).to(self.device).eval()

    @torch.no_grad()
    def score(self, texts, images):
        """images: PIL lista None elemekkel; None helyén 0 (a maszk úgyis hiányzónak jelöli)."""
        out = np.zeros((len(texts), 1), dtype=np.float32)
        idx = [i for i, im in enumerate(images) if im is not None]
        if idx:
            inp = self.proc(text=[texts[i] for i in idx], images=[images[i] for i in idx],
                            return_tensors="pt", padding=True, truncation=True).to(self.device)
            res = self.model(**inp)  # text_embeds / image_embeds: a projektált vektorok (minden verzióban)
            t = torch.nn.functional.normalize(res.text_embeds, dim=-1)
            v = torch.nn.functional.normalize(res.image_embeds, dim=-1)
            out[idx, 0] = (t * v).sum(-1).float().cpu().numpy()
        return out
