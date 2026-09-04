"""Prepare a CSV-labelled retinal dataset without moving the source files."""
import argparse, shutil
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from drd import CLASS_NAMES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", required=True)
    p.add_argument("--labels-csv", required=True)
    p.add_argument("--output-dir", default="data/processed")
    p.add_argument("--test-size", type=float, default=0.10)
    p.add_argument("--validation-size", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    images_dir, output = Path(args.images_dir), Path(args.output_dir)
    df = pd.read_csv(args.labels_csv)
    required = {"image", "level"}
    if not required.issubset(df.columns): raise ValueError(f"CSV must contain columns: {sorted(required)}")
    df["level"] = df["level"].astype(int)
    df = df[df.level.between(0, len(CLASS_NAMES)-1)].copy()
    train_val, test = train_test_split(df, test_size=args.test_size, random_state=args.seed, stratify=df.level)
    val_fraction_of_trainval = args.validation_size / (1 - args.test_size)
    train, val = train_test_split(train_val, test_size=val_fraction_of_trainval, random_state=args.seed, stratify=train_val.level)
    splits = {"train": train, "validation": val, "test": test}
    missing = 0
    for split, part in splits.items():
        for _, row in part.iterrows():
            src = next((images_dir / f"{row.image}{ext}" for ext in [".jpeg", ".jpg", ".png"] if (images_dir / f"{row.image}{ext}").exists()), None)
            if src is None: missing += 1; continue
            dest = output / split / CLASS_NAMES[int(row.level)] / src.name
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dest)
    print(f"Prepared {len(df)-missing} images; missing={missing}")
    for split, part in splits.items(): print(split, part.level.value_counts().sort_index().to_dict())

if __name__ == "__main__": main()
