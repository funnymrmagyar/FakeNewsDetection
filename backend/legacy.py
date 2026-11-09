"""A régi (WELFake BERT + ResNet18, 0,5/0,5 late fusion) modellek. Csak akkor fut,
ha még nincs betanított fusion_head.pt - így az app a Kaggle-tanítás előtt is demózható."""
import torch
import torch.nn as nn
from torchvision import models, transforms
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


class TextClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(getattr(out, "pooler_output", out.last_hidden_state[:, 0, :]))


class LegacyPipeline:
    engine = "legacy"

    def __init__(self, model_dir, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.text_model = TextClassifier().to(device)
        sd = torch.load(model_dir / "text_only_bert.pt", map_location=device)
        try:
            self.text_model.load_state_dict(sd)
        except Exception:
            self.text_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)
            self.text_model.load_state_dict(sd)
        self.text_model.eval()
        m = models.resnet18(weights=None)
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, 2))
        m.load_state_dict(torch.load(model_dir / "best_image_model.pt", map_location=device))
        self.image_model = m.to(device).eval()
        self.tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def info(self):
        return {"engine": self.engine, "features": ["text_bert(WELFake)", "img_resnet18"], "fusion": "late 0.5/0.5"}

    @torch.no_grad()
    def predict(self, text, image):
        probs = []
        if text:
            enc = self.tokenizer(text, truncation=True, padding="max_length", max_length=128,
                                 return_tensors="pt").to(self.device)
            out = self.text_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            probs.append(torch.softmax(getattr(out, "logits", out), dim=1)[0])
        if image is not None:
            probs.append(torch.softmax(self.image_model(self.tf(image).unsqueeze(0).to(self.device)), dim=1)[0])
        p = sum(probs) / len(probs)
        return {"prob_fake": float(p[1]), "contributions": [], "caption": None}
