import torch
import torch.nn as nn


class FusionHead(nn.Module):
    """Konkatenációs fúzió előre kiszámolt embeddingekre.

    dims: az ágak dimenziói, pl. [768] (csak szöveg), [768, 768] (BERT + ViT),
          [768, 768, 768] (BERT + ViT + BLIP-aláírás).
    Minden ághoz tartozik egy tanult hiány-vektor: ha egy mintánál hiányzik
    a modalitás (mask == False), az ág helyére ez kerül, nem nullák.
    """

    def __init__(self, dims, hidden=256, dropout=0.3, num_classes=2):
        super().__init__()
        self.dims = list(dims)
        self.missing = nn.ParameterList([nn.Parameter(torch.zeros(d)) for d in self.dims])
        self.net = nn.Sequential(
            nn.Linear(sum(self.dims), hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def fuse(self, feats, masks=None):
        parts = []
        for i, f in enumerate(feats):
            if masks is not None and masks[i] is not None:
                m = masks[i].bool().unsqueeze(1)
                f = torch.where(m, f, self.missing[i].expand_as(f))
            parts.append(f)
        return torch.cat(parts, dim=1)

    def forward(self, feats, masks=None):
        return self.net(self.fuse(feats, masks))

    def forward_concat(self, x):
        """A már összefűzött vektorra (SHAP ezt hívja)."""
        return self.net(x)


# --- Régi, end-to-end kísérleti modell (a backend nem használja; referenciaként maradt) ---
class RobustMultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision import models
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        resnet = models.resnet18(weights=None)
        resnet.fc = nn.Identity()
        self.resnet = resnet
        self.missing_img_feat = nn.Parameter(torch.randn(1, 512))
        self.classifier = nn.Sequential(
            nn.Linear(768 + 512, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2))

    def forward(self, input_ids=None, attention_mask=None, image_tensor=None, has_text=None, has_image=None):
        device = next(self.parameters()).device
        batch_size = input_ids.size(0) if input_ids is not None else image_tensor.size(0)
        if has_text is None:
            has_text = torch.full((batch_size,), input_ids is not None, dtype=torch.bool, device=device)
        if has_image is None:
            has_image = torch.full((batch_size,), image_tensor is not None, dtype=torch.bool, device=device)
        if input_ids is not None:
            text_feat = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        else:
            text_feat = torch.zeros(batch_size, 768, device=device)
        text_feat = torch.where(has_text.unsqueeze(1), text_feat, torch.zeros_like(text_feat))
        miss = self.missing_img_feat.expand(batch_size, -1)
        img_feat = self.resnet(image_tensor) if image_tensor is not None else miss
        img_feat = torch.where(has_image.unsqueeze(1), img_feat, miss)
        return self.classifier(torch.cat([text_feat, img_feat], dim=1))
