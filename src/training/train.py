"""Egy kísérlet (a terv M1-M8 sorai) betanítása több seeddel, előre kiszámolt embeddingeken.

    python -m src.training.train --name M6_bert_vit --features text_bert img_vit \
        --emb-dir /kaggle/working/emb --splits /kaggle/working/data/splits.csv --out results

Kimenet a --out mappában:
    preds/<name>_seed<s>.csv   (id, label, prob_fake) a teszten -> szignifikancia-tesztek
    models/<name>_seed<s>.pt   (fej súlyai + metaadat) -> backend
    summary.csv                (soronként egy futás; aggregálás: src.eval.report)
"""
import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight

from src.eval.metrics import compute
from src.models.classifier import FusionHead

IMG_FEATURES = {"img_vit", "img_resnet18", "caption_bert", "clip_score"}


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def load(emb_dir, splits_csv, features):
    df = pd.read_csv(splits_csv, dtype={"id": str})
    ids = np.load(os.path.join(emb_dir, "ids.npy"), allow_pickle=True)
    if len(ids) != len(df) or not (ids.astype(str) == df["id"].to_numpy()).all():
        raise SystemExit("Az embeddingek sorrendje nem egyezik a splits.csv-vel. Futtasd újra az extract-ot.")
    X = [torch.from_numpy(np.load(os.path.join(emb_dir, f"{f}.npy"))).float() for f in features]
    ok_path = os.path.join(emb_dir, "img_ok.npy")
    img_ok = torch.from_numpy(np.load(ok_path)) if os.path.exists(ok_path) else torch.ones(len(df), dtype=torch.bool)
    ones = torch.ones(len(df), dtype=torch.bool)
    M = [img_ok.bool() if f in IMG_FEATURES else ones for f in features]
    return df, X, M


def modality_dropout(masks, p):
    """Soronként p valószínűséggel egy véletlen modalitást hiányzónak jelöl (soha nem az összeset).
    Így a tanult hiány-vektorok is tanulnak, és a webapp csak-szöveg / csak-kép bemenettel is működik;
    a modalitás-szintű SHAP is erre a \"hiányzó\" állapotra épül."""
    n = len(masks)
    if n < 2 or p <= 0:
        return masks
    B = masks[0].shape[0]
    hit = torch.rand(B, device=masks[0].device) < p
    which = torch.randint(0, n, (B,), device=masks[0].device)
    return [m & ~(hit & (which == i)) for i, m in enumerate(masks)]


def batches(idx, bs, shuffle):
    idx = idx[torch.randperm(len(idx))] if shuffle else idx
    for i in range(0, len(idx), bs):
        yield idx[i:i + bs]


@torch.no_grad()
def predict(model, X, M, idx, dev, bs=2048):
    model.eval()
    out = []
    for b in batches(idx, bs, False):
        feats = [x[b].to(dev) for x in X]
        masks = [m[b].to(dev) for m in M]
        out.append(torch.softmax(model(feats, masks), dim=1)[:, 1].cpu())
    return torch.cat(out).numpy()


def run(args, seed, df, X, M, dev):
    set_seed(seed)
    y = torch.tensor(df["label"].to_numpy())
    tr, va, te = (torch.tensor(np.where(df["split"] == s)[0]) for s in ("train", "val", "test"))
    model = FusionHead([x.shape[1] for x in X], hidden=args.hidden, dropout=args.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    weight = None
    if args.class_weight:
        w = compute_class_weight("balanced", classes=np.array([0, 1]), y=y[tr].numpy())
        weight = torch.tensor(w, dtype=torch.float32, device=dev)
    loss_fn = nn.CrossEntropyLoss(weight=weight)

    best_f1, best_state, bad = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for b in batches(tr, args.batch_size, True):
            feats = [x[b].to(dev) for x in X]
            masks = modality_dropout([m[b].to(dev) for m in M], args.modality_dropout)
            loss = loss_fn(model(feats, masks), y[b].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
        f1 = compute(y[va].numpy(), predict(model, X, M, va, dev))["f1"]
        if f1 > best_f1:
            best_f1, best_state, bad = f1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)

    prob = predict(model, X, M, te, dev)
    res = compute(y[te].numpy(), prob)
    os.makedirs(os.path.join(args.out, "preds"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "models"), exist_ok=True)
    pd.DataFrame({"id": df["id"].to_numpy()[te.numpy()], "label": y[te].numpy(), "prob_fake": prob}) \
        .to_csv(os.path.join(args.out, "preds", f"{args.name}_seed{seed}.csv"), index=False)
    torch.save({"state_dict": best_state, "features": args.features, "dims": model.dims,
                "hidden": args.hidden, "dropout": args.dropout, "val_f1": best_f1, "test": res,
                "name": args.name, "seed": seed, "modality_dropout": args.modality_dropout},
               os.path.join(args.out, "models", f"{args.name}_seed{seed}.pt"))
    return {"name": args.name, "features": "+".join(args.features), "seed": seed, "epochs": epoch,
            "val_f1": best_f1, **{k: v for k, v in res.items() if k != "cm"}, "cm": json.dumps(res["cm"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)),
                    help="10 seed: n=5-nél a Wilcoxon-teszt legkisebb elérhető p-értéke 0,0625, az nem szignifikáns.")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--modality-dropout", type=float, default=0.15)
    ap.add_argument("--class-weight", action="store_true", help="Kiegyensúlyozatlan adathoz (FakeNewsNet).")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df, X, M = load(args.emb_dir, args.splits, args.features)
    rows = []
    for s in args.seeds:
        r = run(args, s, df, X, M, dev)
        print(f"{args.name} seed={s}: acc={r['acc']:.4f} f1={r['f1']:.4f} auc={r['auc']:.4f} (epoch {r['epochs']})")
        rows.append(r)
    path = os.path.join(args.out, "summary.csv")
    pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    f1 = np.array([r["f1"] for r in rows])
    print(f"==> {args.name}: F1 = {f1.mean():.4f} ± {f1.std(ddof=1) if len(f1) > 1 else 0:.4f}")


if __name__ == "__main__":
    main()
