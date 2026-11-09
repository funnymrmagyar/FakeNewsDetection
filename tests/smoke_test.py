"""Gyors ellenőrzés kitalált adaton: a teljes pipeline (tanítás, late fusion, szignifikancia,
riport, SHAP) lefut-e hiba nélkül, és helyesek-e a kulcs-tulajdonságok.
Nem tölt le előtanított modellt, nem kell hozzá GPU. Kb. 1-2 perc.

Futtatás a projekt gyökeréből:
    backend\\venv\\Scripts\\python -m tests.smoke_test
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(f"[{'OK ' if cond else 'HIBA'}] {name} {detail}")


def run(*args):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}  # Windows-konzol: ékezetek
    r = subprocess.run([sys.executable, "-m", *args], cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env)
    ok = r.returncode == 0
    check(" ".join(args[:1]), ok)
    if not ok:
        print(r.stdout[-2000:], r.stderr[-3000:], sep="\n")
    return ok


def make_data(d, n=600, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    split = rng.choice(["train", "val", "test"], n, p=[0.7, 0.15, 0.15])
    ids = np.array([f"id{i:05d}" for i in range(n)])
    pd.DataFrame({"id": ids, "text": [f"title {i}" for i in range(n)], "label": y,
                  "img_path": [f"{i}.jpg" for i in range(n)], "split": split}).to_csv(f"{d}/splits.csv", index=False)
    emb = f"{d}/emb"
    os.makedirs(emb)
    np.save(f"{emb}/ids.npy", ids)
    # erős jel a szövegben, gyengébb a képben, gyenge az aláírásban
    for name, dim, strength in [("text_bert", 768, 1.0), ("img_vit", 768, 0.5),
                                ("img_resnet18", 512, 0.3), ("caption_bert", 768, 0.2), ("clip_score", 1, 0.5)]:
        x = rng.normal(size=(n, dim)).astype(np.float32)
        x[:, :8] += strength * (2 * y[:, None] - 1)
        np.save(f"{emb}/{name}.npy", x)
    ok = np.ones(n, bool)
    ok[rng.choice(n, 30, replace=False)] = False
    np.save(f"{emb}/img_ok.npy", ok)
    return emb


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"python {sys.version.split()[0]}, torch {torch.__version__}")
    with tempfile.TemporaryDirectory() as d:
        emb, out = make_data(d), f"{d}/results"
        common = ["--emb-dir", emb, "--splits", f"{d}/splits.csv", "--out", out,
                  "--seeds", "42", "43", "44", "45", "46", "47", "--epochs", "8", "--patience", "3"]
        run("src.training.train", "--name", "M1_text", "--features", "text_bert", *common)
        run("src.training.train", "--name", "M2_resnet", "--features", "img_resnet18", *common)
        run("src.eval.late_fusion", "--a", "M1_text", "--b", "M2_resnet", "--name", "M4_late_fusion",
            "--out", out, "--seeds", "42", "43", "44", "45", "46", "47")
        run("src.training.train", "--name", "M6_bert_vit", "--features", "text_bert", "img_vit", *common)
        run("src.training.train", "--name", "M7_bert_blip_vit", "--features", "text_bert", "img_vit",
            "caption_bert", *common)
        run("src.training.train", "--name", "M8", "--features", "text_bert", "img_vit", "caption_bert",
            "clip_score", *common)
        run("src.stats.significance", "--out", out, "--pairs", "M6_bert_vit:M1_text", "M7_bert_blip_vit:M6_bert_vit",
            "M6_bert_vit:M4_late_fusion")
        run("src.eval.report", "--out", out, "--roc", "M1_text", "M6_bert_vit", "--cm", "M6_bert_vit")
        run("src.explain.shap_analysis", "--ckpt", f"{out}/models/M7_bert_blip_vit_seed42.pt",
            "--emb-dir", emb, "--splits", f"{d}/splits.csv", "--out", out)

        s = pd.read_csv(f"{out}/summary.csv")
        f1 = s.groupby("name")["f1"].mean()
        print(f1.round(3).to_string())
        check("a szöveges modell tanul (F1 > 0.8)", f1.get("M1_text", 0) > 0.8, f"{f1.get('M1_text', 0):.3f}")
        for f in ["significance.csv", "table_main.csv", "roc.png", "cm_M6_bert_vit.png",
                  "shap/M7_bert_blip_vit_summary.csv", "shap/M7_bert_blip_vit_modality.png"]:
            check(f"kimenet: {f}", os.path.exists(f"{out}/{f}"))

        # Shapley: hatékonyság (sum phi + base == prob) és hiányzó modalitás hozzájárulása 0
        from src.explain.shap_analysis import contributions_for_api, load_head, modality_shapley
        head, ck = load_head(f"{out}/models/M7_bert_blip_vit_seed42.pt")
        X = [torch.from_numpy(np.load(f"{emb}/{f}.npy")[:50]) for f in ck["features"]]
        masks = [torch.ones(50, dtype=torch.bool), torch.arange(50) % 5 != 0, torch.ones(50, dtype=torch.bool)]
        phi, base, prob = modality_shapley(head, X, masks)
        check("Shapley: sum(phi) + base == P(fake)", torch.allclose(phi.sum(1) + base, prob, atol=1e-5))
        check("Shapley: hiányzó kép hozzájárulása 0", torch.allclose(phi[~masks[1], 1], torch.zeros(10), atol=1e-6))
        c = contributions_for_api(phi[0].numpy(), ["szöveg", "kép", "képaláírás"])
        check("API: részesedések összege 100%", abs(sum(x["share"] for x in c) - 100) < 0.5)
        print("Shapley-részesedés (átlag |phi|):",
              dict(zip(["szöveg", "kép", "aláírás"], phi.abs().mean(0).numpy().round(4))))

    bad = [n for n, ok in results if not ok]
    print("\n" + ("MINDEN RENDBEN" if not bad else f"{len(bad)} HIBA: {bad}"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
