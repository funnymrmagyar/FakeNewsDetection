"""Szignifikancia-tesztek modellpárokra (a terv 3. fázisa).

- McNemar (egzakt, binomiális): két modell ugyanazon a teszthalmazon, seedenként.
- Páros t-teszt és Wilcoxon: a seedenkénti F1-értékekre (párosítás seed szerint).

    python -m src.stats.significance --out results \
        --pairs M6_bert_vit:M1_text M6_bert_vit:M5_bert_resnet M3_vit:M2_resnet M7_bert_blip_vit:M6_bert_vit
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_rel, wilcoxon


def mcnemar_exact(y, pa, pb):
    ca, cb = pa == y, pb == y
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue
    return b, c, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--pairs", nargs="+", required=True, help="A:B formában; A az új, B a viszonyítási modell.")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    summ = pd.read_csv(os.path.join(args.out, "summary.csv")).drop_duplicates(["name", "seed"], keep="last")
    rows = []
    for pair in args.pairs:
        a, b = pair.split(":")
        fa = summ[summ.name == a].set_index("seed")["f1"]
        fb = summ[summ.name == b].set_index("seed")["f1"]
        seeds = sorted(set(fa.index) & set(fb.index))
        if not seeds:
            print(f"Kihagyva (nincs közös seed): {pair}")
            continue
        mc = []
        for s in seeds:
            A = pd.read_csv(os.path.join(args.out, "preds", f"{a}_seed{s}.csv"), dtype={"id": str})
            B = pd.read_csv(os.path.join(args.out, "preds", f"{b}_seed{s}.csv"), dtype={"id": str})
            m = A.merge(B, on=["id", "label"], suffixes=("_a", "_b"))
            mc.append(mcnemar_exact(m["label"].to_numpy(), (m["prob_fake_a"] >= .5).to_numpy().astype(int),
                                    (m["prob_fake_b"] >= .5).to_numpy().astype(int)))
        x, z = fa[seeds].to_numpy(), fb[seeds].to_numpy()
        t_p = ttest_rel(x, z).pvalue if len(seeds) > 1 else np.nan
        w_p = wilcoxon(x, z).pvalue if len(seeds) > 1 and np.any(x != z) else np.nan
        mc_p = np.array([p for _, _, p in mc])
        rows.append({
            "A": a, "B": b, "seedek": len(seeds),
            "F1 A (átlag)": x.mean(), "F1 B (átlag)": z.mean(), "ΔF1": (x - z).mean(),
            "t-teszt p": t_p, "Wilcoxon p": w_p,
            "McNemar p (seed 1)": mc_p[0], "McNemar p (medián)": np.median(mc_p),
            "McNemar szign. seedek": f"{(mc_p < args.alpha).sum()}/{len(mc_p)}",
            "csak A jó / csak B jó (seed 1)": f"{mc[0][0]} / {mc[0][1]}",
        })
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.out, "significance.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(res.round(5).to_string(index=False))
    if len(rows) and rows[0]["seedek"] < 6:
        print("Figyelem: 6-nál kevesebb seednél a Wilcoxon-teszt nem tud 0,05 alá menni.")


if __name__ == "__main__":
    main()
