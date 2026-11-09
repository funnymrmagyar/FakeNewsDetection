"""3. fázis (RQ3): melyik modalitás mennyivel járul hozzá a döntéshez.

Módszer: pontos Shapley-értékek modalitás-szinten (a SHAP elméleti alapja), ahol a
"játékosok" a modalitások (szöveg / kép / képaláírás), a hiányzó modalitás helyére
pedig a modell tanult hiány-vektora kerül (ezt a modality dropout tanítja).
n <= 3 modalitásnál ez 2^n kiértékelés mintánként, közelítés nélkül.
Opcionálisan (--per-dim) a `shap` könyvtár GradientExplainer-ével dimenziónkénti
SHAP is számolható, modalitásonként összegezve (a két módszer összevetése a dolgozatba).

    python -m src.explain.shap_analysis --ckpt results/models/M7_bert_blip_vit_seed42.pt \
        --emb-dir /kaggle/working/emb --splits /kaggle/working/data/splits.csv --out results
"""
import argparse
import itertools
import math
import os

import numpy as np
import pandas as pd
import torch

from src.models.classifier import FusionHead
from src.models.encoders import MODALITY_NAMES


def load_head(ckpt_path, device="cpu"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    head = FusionHead(ck["dims"], hidden=ck["hidden"], dropout=ck["dropout"]).to(device)
    head.load_state_dict(ck["state_dict"])
    return head.eval(), ck


@torch.no_grad()
def modality_shapley(head, feats, masks=None):
    """feats: n tenzor (B x d_i); masks: n bool tenzor (B,) vagy None (= minden jelen).
    Vissza: phi (B x n) a P(fake)-re, base (B,) = P(fake) minden modalitás hiányában,
    prob (B,) = a teljes bemenet P(fake)-je. sum(phi) + base == prob (hatékonysági tulajdonság).
    Egy eleve hiányzó modalitás hozzájárulása 0."""
    n, B, dev = len(feats), feats[0].shape[0], feats[0].device
    present = masks if masks is not None else [torch.ones(B, dtype=torch.bool, device=dev)] * n

    def v(S):
        m = [present[i] & (i in S) for i in range(n)]
        return torch.softmax(head(feats, m), dim=1)[:, 1]

    cache = {S: v(set(S)) for r in range(n + 1) for S in itertools.combinations(range(n), r)}
    phi = torch.zeros(B, n, device=dev)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for r in range(n):
            w = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
            for S in itertools.combinations(others, r):
                phi[:, i] += w * (cache[tuple(sorted(S + (i,)))] - cache[S])
    return phi, cache[()], cache[tuple(range(n))]


def contributions_for_api(phi_row, names):
    """Egy minta Shapley-értékeiből a frontendnek szánt lista (százalékos részesedés + irány)."""
    tot = float(np.abs(phi_row).sum()) or 1.0
    return [{"modality": nm, "value": round(float(p), 4), "share": round(100 * abs(float(p)) / tot, 1),
             "direction": "fake" if p > 0 else "real"} for nm, p in zip(names, phi_row)]


def per_dim_shap(head, X, idx, n_bg=200, n_explain=1000, seed=0):
    import shap
    rng = np.random.default_rng(seed)
    full = torch.cat([x for x in X], dim=1)
    bg = full[rng.choice(len(full), n_bg, replace=False)]
    ex = full[idx[:n_explain]]

    class Wrap(torch.nn.Module):
        def __init__(self, h):
            super().__init__()
            self.h = h

        def forward(self, z):
            return torch.softmax(self.h.forward_concat(z), dim=1)[:, 1:2]

    sv = shap.GradientExplainer(Wrap(head), bg).shap_values(ex)
    sv = np.asarray(sv).reshape(len(ex), -1)
    bounds = np.cumsum([0] + [x.shape[1] for x in X])
    return np.stack([np.abs(sv[:, a:b]).sum(1) for a, b in zip(bounds[:-1], bounds[1:])], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--per-dim", action="store_true")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.training.train import load

    head, ck = load_head(args.ckpt)
    feats_names = ck["features"]
    names = [MODALITY_NAMES[f] for f in feats_names]
    df, X, M = load(args.emb_dir, args.splits, feats_names)
    te = np.where(df["split"] == "test")[0]
    phis, probs = [], []
    for a in range(0, len(te), 4096):
        b = torch.tensor(te[a:a + 4096])
        phi, base, prob = modality_shapley(head, [x[b] for x in X], [m[b] for m in M])
        phis.append(phi.numpy())
        probs.append(prob.numpy())
    phi, prob = np.concatenate(phis), np.concatenate(probs)

    out_dir = os.path.join(args.out, "shap")
    os.makedirs(out_dir, exist_ok=True)
    tag = ck.get("name", os.path.splitext(os.path.basename(args.ckpt))[0])
    per = pd.DataFrame(phi, columns=[f"phi_{n}" for n in names])
    per.insert(0, "id", df["id"].to_numpy()[te])
    per.insert(1, "label", df["label"].to_numpy()[te])
    per.insert(2, "text", df["text"].to_numpy()[te])
    per["prob_fake"] = prob
    per["pred"] = (prob >= 0.5).astype(int)
    per["dominant"] = [names[i] for i in np.abs(phi).argmax(1)]
    per.to_csv(os.path.join(out_dir, f"{tag}_per_sample.csv"), index=False)

    share = np.abs(phi) / np.clip(np.abs(phi).sum(1, keepdims=True), 1e-9, None)
    summ = pd.DataFrame({"modalitás": names, "átlag |phi|": np.abs(phi).mean(0),
                         "átlagos részesedés %": 100 * share.mean(0),
                         "domináns a minták %-ában": [100 * (per["dominant"] == n).mean() for n in names]})
    if args.per_dim:
        pdm = per_dim_shap(head, X, te)
        summ["GradientExplainer átlag |SHAP|"] = pdm.mean(0)
    summ.to_csv(os.path.join(out_dir, f"{tag}_summary.csv"), index=False)
    print(summ.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.barh(names, summ["átlagos részesedés %"], color="#4C78A8")
    ax.set_xlabel("Átlagos hozzájárulás a döntéshez (%)")
    ax.set_title(f"Modalitás-hozzájárulás – {tag}")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{tag}_modality.png"), dpi=200)

    # példák a dolgozatba: helyes döntések, ahol a kép ill. a szöveg dominált, és a legrosszabb tévedések
    correct = per[per["pred"] == per["label"]]
    ex = pd.concat([
        correct[correct["dominant"] == "kép"].assign(miért="helyes, a kép döntött").head(3),
        correct[correct["dominant"] == "szöveg"].assign(miért="helyes, a szöveg döntött").head(3),
        per[per["pred"] != per["label"]].assign(conf=lambda d: (d["prob_fake"] - 0.5).abs())
           .sort_values("conf", ascending=False).drop(columns="conf").assign(miért="magabiztos tévedés").head(10),
    ])
    ex.to_csv(os.path.join(out_dir, f"{tag}_examples.csv"), index=False)
    print(f"Mentve: {out_dir}/")


if __name__ == "__main__":
    main()
