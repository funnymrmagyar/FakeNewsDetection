"""0. fázis: Fakeddit TSV + Fakeddit-50k képek összekötése, címkék, rétegzett felosztás.

Kaggle-en:
    python -m src.data.prepare \
        --tsv-dir /kaggle/input/<fakeddit-dataset>/multimodal_only_samples \
        --img-dir /kaggle/input/fakeddit-50k/fakeddit_small_img \
        --out /kaggle/working/data

Kimenet: splits.csv (id, text, label, img_path, split) és stats.csv.
Címkekonvenció a projektben: 0 = real, 1 = fake.
"""
import argparse
import glob
import os

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


def read_tsvs(tsv_dir):
    files = sorted(glob.glob(os.path.join(tsv_dir, "*.tsv")))
    if not files:
        raise FileNotFoundError(f"Nincs .tsv fájl itt: {tsv_dir}")
    frames = []
    for f in files:
        df = pd.read_csv(f, sep="\t", dtype={"id": str}, on_bad_lines="skip", low_memory=False)
        print(f"  {os.path.basename(f)}: {len(df):,} sor")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def detect_fake_value(df):
    """A Fakeddit 2_way_label kódolását az r/theonion (szatíra = fake) posztjaiból olvassa ki."""
    if "subreddit" not in df.columns:
        return None
    onion = df[df["subreddit"].astype(str).str.lower() == "theonion"]
    if len(onion) < 20:
        return None
    return int(onion["2_way_label"].mode().iloc[0])


def image_ok(path):
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv-dir", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fake-value", type=int, default=None,
                    help="Melyik 2_way_label érték jelenti a fake-et. Alapból az r/theonion alapján derül ki.")
    ap.add_argument("--max-samples", type=int, default=None, help="Opcionális lemintavételezés (kiegyensúlyozva).")
    ap.add_argument("--check-images", action="store_true", help="Minden képet megnyit, a sérülteket kidobja (lassabb).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("TSV-k beolvasása...")
    df = read_tsvs(args.tsv_dir)

    fake_value = args.fake_value if args.fake_value is not None else detect_fake_value(df)
    if fake_value is None:
        raise SystemExit("Nem tudtam eldönteni a címkekódolást. Add meg: --fake-value 0 vagy 1.")
    print(f"Címkekódolás: 2_way_label == {fake_value} -> fake (1), a másik -> real (0)")

    df = df[["id", "clean_title", "2_way_label"]].dropna()
    df = df.rename(columns={"clean_title": "text"})
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    df["label"] = (df["2_way_label"].astype(int) == fake_value).astype(int)
    df = df.drop(columns=["2_way_label"]).drop_duplicates("id")

    # csak azok, amelyekhez van kép
    files = {os.path.splitext(f)[0]: f for f in os.listdir(args.img_dir)}
    df = df[df["id"].isin(files)].copy()
    df["img_path"] = df["id"].map(lambda i: os.path.join(args.img_dir, files[i]))
    print(f"Képpel rendelkező minták: {len(df):,}")

    # adatszivárgás ellen: azonos cím csak egyszer, ellentmondó címkéjűek kidobva
    conflict = df.groupby("text")["label"].nunique()
    df = df[~df["text"].isin(conflict[conflict > 1].index)]
    before = len(df)
    df = df.drop_duplicates("text")
    print(f"Duplikált címek eltávolítva: {before - len(df):,}")

    if args.check_images:
        ok = df["img_path"].map(image_ok)
        print(f"Sérült képek: {(~ok).sum():,}")
        df = df[ok]

    if args.max_samples and len(df) > args.max_samples:
        per = args.max_samples // 2
        df = (df.groupby("label", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), per), random_state=args.seed)))

    train, rest = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=args.seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=args.seed)
    train, val, test = train.assign(split="train"), val.assign(split="val"), test.assign(split="test")
    out = pd.concat([train, val, test]).reset_index(drop=True)
    out.to_csv(os.path.join(args.out, "splits.csv"), index=False)

    stats = (out.assign(label=out["label"].map({0: "real", 1: "fake"}))
                .pivot_table(index="split", columns="label", values="id", aggfunc="count")
                .reindex(["train", "val", "test"]))
    stats["total"] = stats.sum(axis=1)
    stats["fake_ratio"] = (stats["fake"] / stats["total"]).round(3)
    stats["mean_title_words"] = out.groupby("split")["text"].apply(lambda s: s.str.split().str.len().mean()).round(1)
    stats.to_csv(os.path.join(args.out, "stats.csv"))
    print(stats.to_string())
    print(f"Mentve: {args.out}/splits.csv")


if __name__ == "__main__":
    main()
