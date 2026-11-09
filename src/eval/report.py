"""Eredménytáblák és ábrák a dolgozathoz a results/ mappából.

    python -m src.eval.report --out results --roc M1_text M3_vit M6_bert_vit M7_bert_blip_vit \
        --cm M6_bert_vit M7_bert_blip_vit

Kimenet: table_main.csv / .md (átlag ± szórás), roc.png, cm_<modell>.png.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--roc", nargs="*", default=[])
    ap.add_argument("--cm", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=42, help="Ennek a seednek a predikcióiból készülnek az ábrák.")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    s = pd.read_csv(os.path.join(args.out, "summary.csv")).drop_duplicates(["name", "seed"], keep="last")
    metrics = ["acc", "precision", "recall", "f1", "auc"]
    g = s.groupby("name")
    tab = pd.DataFrame({"jellemzők": g["features"].first(), "seedek": g.size()})
    for m in metrics:
        tab[m] = g[m].mean().map("{:.4f}".format) + " ± " + g[m].std(ddof=1).fillna(0).map("{:.4f}".format)
    tab = tab.loc[sorted(tab.index)]
    tab.to_csv(os.path.join(args.out, "table_main.csv"))
    try:
        with open(os.path.join(args.out, "table_main.md"), "w", encoding="utf-8") as f:
            f.write(tab.to_markdown())
    except ImportError:
        print("(a tabulate nincs telepítve, table_main.md kimarad)")
    print(tab.to_string())

    def preds(name):
        return pd.read_csv(os.path.join(args.out, "preds", f"{name}_seed{args.seed}.csv"))

    if args.roc:
        fig, ax = plt.subplots(figsize=(5, 4.5))
        for name in args.roc:
            p = preds(name)
            fpr, tpr, _ = roc_curve(p["label"], p["prob_fake"])
            auc = s[(s.name == name) & (s.seed == args.seed)]["auc"].iloc[0]
            ax.plot(fpr, tpr, label=f"{name} (AUC {auc:.3f})")
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=0.8)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "roc.png"), dpi=200)

    for name in args.cm:
        cm = np.array(json.loads(s[(s.name == name) & (s.seed == args.seed)]["cm"].iloc[0]))
        fig, ax = plt.subplots(figsize=(3.6, 3.2))
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["real", "fake"])
        ax.set_yticks([0, 1], ["real", "fake"])
        ax.set_xlabel("Jósolt")
        ax.set_ylabel("Valós")
        ax.set_title(name, fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"cm_{name}.png"), dpi=200)
    print(f"Mentve: {args.out}/")


if __name__ == "__main__":
    main()
