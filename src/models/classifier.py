import torch
import torch.nn as nn
from transformers import AutoModel
from torchvision import models

class RobustMultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Szöveges ág (BERT)
        self.bert = AutoModel.from_pretrained("bert-base-uncased")

        # 2. Képi ág (ResNet18)
        resnet = models.resnet18(weights=None)
        resnet.fc = nn.Identity()  # 512-dim kimenet, nincs saját osztályozó réteg
        self.resnet = resnet

        # Tanult hiány-vektor a képhez
        self.missing_img_feat = nn.Parameter(torch.randn(1, 512))

        # Osztályozó fej
        self.classifier = nn.Sequential(
            nn.Linear(768 + 512, 256),  # index 0
            nn.LayerNorm(256),          # index 1
            nn.ReLU(),                  # index 2
            nn.Dropout(0.3),            # index 3
            nn.Linear(256, 2)           # index 4
        )

    def forward(self, input_ids=None, attention_mask=None, image_tensor=None, has_text=None, has_image=None):
        device = next(self.parameters()).device
        
        # 1. Batch méret dinamikus meghatározása a meglévő bemenetből
        if input_ids is not None:
            batch_size = input_ids.size(0)
        elif image_tensor is not None:
            batch_size = image_tensor.size(0)
        else:
            raise ValueError("❌ Legalább egy bemenetet (szöveg vagy kép) kötelező megadni!")

        # 2. Boolean maszkok automatikus legenerálása, ha API-ból hívjuk (None esetén)
        if has_text is None:
            has_text = torch.tensor([input_ids is not None] * batch_size, dtype=torch.bool, device=device)
        if has_image is None:
            has_image = torch.tensor([image_tensor is not None] * batch_size, dtype=torch.bool, device=device)

        # --- SZÖVEGES ÁG FELDOLGOZÁSA ---
        if input_ids is not None and attention_mask is not None:
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            text_feat = bert_out.pooler_output
        else:
            # Ha nincs szöveges bemenet (API hívásnál), csupa 0-t küldünk be
            text_feat = torch.zeros(batch_size, 768, device=device)
            
        # Extra biztosítás: ha a DataLoader dummy tenzort küldött (de has_text = False)
        has_text_expanded = has_text.unsqueeze(1)
        text_feat = torch.where(has_text_expanded, text_feat, torch.zeros_like(text_feat))

        # --- KÉPI ÁG FELDOLGOZÁSA ---
        if image_tensor is not None:
            img_feat = self.resnet(image_tensor)
        else:
            # Ha nincs képi bemenet, rögtön a hiány-vektort alkalmazzuk
            img_feat = self.missing_img_feat.expand(batch_size, -1)
            
        # A te eredeti logikád: ha van kép, azt használjuk, ha nincs, a hiányvektort
        # (ha DataLoader-ből jön a has_image maszk)
        has_image_expanded = has_image.unsqueeze(1)
        img_feat = torch.where(has_image_expanded, img_feat, self.missing_img_feat.expand(batch_size, -1))

        # --- ÖSSZEFŰZÉS ÉS OSZTÁLYOZÁS ---
        fused = torch.cat([text_feat, img_feat], dim=1)
        return self.classifier(fused)