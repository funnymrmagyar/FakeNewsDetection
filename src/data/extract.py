"""1. fázis: embeddingek kinyerése egyszer, mentés .npy-ba. Utána minden kísérlet percek alatt fut.

    python -m src.data.extract --splits /kaggle/working/data/splits.csv \
        --out /kaggle/working/emb --which text_bert img_vit img_resnet18

Kimenet: <feature>.npy (N x dim, a splits.csv sorrendjében), img_ok.npy (bool),
ids.npy. Ezt a mappát mentsd el Kaggle-datasetként.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.models.encoders import CaptionGenerator, ClipScorer, ImageEncoder, TextEncoder, load_image


class ImageDS(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = list(paths), transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = load_image(self.paths[i])
        if im is None:
            return torch.zeros(3, 224, 224), False
        return self.transform(im), True


def extract_text(df, out, bs):
    enc = TextEncoder()
    texts = df["text"].tolist()
    chunks = [enc.encode(texts[i:i + bs]) for i in tqdm(range(0, len(texts), bs), desc="text_bert")]
    np.save(os.path.join(out, "text_bert.npy"), np.concatenate(chunks).astype(np.float32))


def extract_image(df, out, kind, bs, workers):
    enc = ImageEncoder(kind)
    dl = DataLoader(ImageDS(df["img_path"], enc.transform), batch_size=bs, num_workers=workers)
    feats, oks = [], []
    for x, ok in tqdm(dl, desc=f"img_{kind}"):
        feats.append(enc.encode_tensors(x))
        oks.append(ok.numpy())
    ok = np.concatenate(oks)
    feats = np.concatenate(feats).astype(np.float32)
    feats[~ok] = 0.0
    np.save(os.path.join(out, f"img_{kind}.npy"), feats)
    np.save(os.path.join(out, "img_ok.npy"), ok)
    print(f"Hibás/hiányzó kép: {(~ok).sum()} / {len(ok)}")


def extract_caption(df, out, bs):
    """3. fázis (RQ2): BLIP-képaláírás -> captions.csv -> BERT [CLS] embedding.
    Hiányzó képnél üres aláírás; ezeket az img_ok maszk jelöli (a train.py hiányzónak veszi)."""
    gen = CaptionGenerator()
    caps = []
    paths = df["img_path"].tolist()
    for i in tqdm(range(0, len(paths), bs), desc="blip"):
        caps += gen.generate([load_image(p) for p in paths[i:i + bs]])
    pd.DataFrame({"id": df["id"], "caption": caps}).to_csv(os.path.join(out, "captions.csv"), index=False)
    del gen
    torch.cuda.empty_cache()
    enc = TextEncoder()
    chunks = [enc.encode([c or "[no image]" for c in caps[i:i + 64]]) for i in range(0, len(caps), 64)]
    np.save(os.path.join(out, "caption_bert.npy"), np.concatenate(chunks).astype(np.float32))


def extract_clip(df, out, bs):
    """M8: CLIP szöveg–kép koszinusz-hasonlóság, N x 1."""
    sc = ClipScorer()
    texts, paths = df["text"].tolist(), df["img_path"].tolist()
    chunks = [sc.score(texts[i:i + bs], [load_image(p) for p in paths[i:i + bs]])
              for i in tqdm(range(0, len(texts), bs), desc="clip")]
    np.save(os.path.join(out, "clip_score.npy"), np.concatenate(chunks).astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--which", nargs="+", default=["text_bert", "img_vit", "img_resnet18"],
                    choices=["text_bert", "img_vit", "img_resnet18", "caption_bert", "clip_score"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.splits, dtype={"id": str})
    np.save(os.path.join(args.out, "ids.npy"), df["id"].to_numpy())
    df.to_csv(os.path.join(args.out, "splits.csv"), index=False)  # az embeddingekkel együtt mentve
    for w in args.which:
        if w == "text_bert":
            extract_text(df, args.out, args.batch_size)
        elif w == "img_vit":
            extract_image(df, args.out, "vit", args.batch_size, args.workers)
        elif w == "img_resnet18":
            extract_image(df, args.out, "resnet18", args.batch_size, args.workers)
        elif w == "caption_bert":
            extract_caption(df, args.out, 16)
        elif w == "clip_score":
            extract_clip(df, args.out, args.batch_size)
    print("Kész:", sorted(os.listdir(args.out)))


if __name__ == "__main__":
    main()
