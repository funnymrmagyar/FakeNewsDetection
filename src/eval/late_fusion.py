"""M4 baseline: a mostani webapp late fusionje (0,5 * P_szöveg + 0,5 * P_kép),
két már lefutott modell teszt-predikcióiból, seedenként.

    python -m src.eval.late_fusion --a M1_text --b M2_resnet --name M4_late_fusion --out results
"""
import argparse
import json
import os

import pandas as pd

from src.eval.metrics import compute


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    args = ap.parse_args()
    rows = []
    for s in args.seeds:
        a = pd.read_csv(os.path.join(args.out, "preds", f"{args.a}_seed{s}.csv"), dtype={"id": str})
        b = pd.read_csv(os.path.join(args.out, "preds", f"{args.b}_seed{s}.csv"), dtype={"id": str})
        m = a.merge(b[["id", "prob_fake"]], on="id", suffixes=("_a", "_b"))
        m["prob_fake"] = 0.5 * m["prob_fake_a"] + 0.5 * m["prob_fake_b"]
        m[["id", "label", "prob_fake"]].to_csv(os.path.join(args.out, "preds", f"{args.name}_seed{s}.csv"), index=False)
        r = compute(m["label"], m["prob_fake"])
        rows.append({"name": args.name, "features": f"late({args.a},{args.b})", "seed": s, "epochs": 0,
                     "val_f1": float("nan"), **{k: v for k, v in r.items() if k != "cm"}, "cm": json.dumps(r["cm"])})
        print(f"{args.name} seed={s}: f1={r['f1']:.4f}")
    path = os.path.join(args.out, "summary.csv")
    pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


if __name__ == "__main__":
    main()
