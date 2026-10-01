import argparse
import csv
import json
import re
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_files(folder):
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def row(dataset, split, sketch_path, photo_path):
    return {
        "dataset": dataset,
        "split": split,
        "sketch_path": str(sketch_path.resolve()),
        "photo_path": str(photo_path.resolve()),
    }


def pair_by_stem(dataset, split, sketch_dir, photo_dir, *, sketch_suffix=""):
    sketches = {}
    for sketch_path in image_files(sketch_dir):
        stem = sketch_path.stem
        if sketch_suffix and stem.endswith(sketch_suffix):
            stem = stem[: -len(sketch_suffix)]
        sketches[stem.lower()] = sketch_path

    pairs = []
    for photo_path in image_files(photo_dir):
        sketch_path = sketches.get(photo_path.stem.lower())
        if sketch_path is not None:
            pairs.append(row(dataset, split, sketch_path, photo_path))
    return pairs


def find_first(root, name):
    for path in root.rglob(name):
        if path.is_dir():
            return path
    return None


def person_face_sketches(raw_root):
    root = raw_root / "person_face_sketches"
    pairs = []
    for split in ("train", "val", "test"):
        photo_dir = root / split / "photos"
        sketch_dir = root / split / "sketches"
        pairs.extend(pair_by_stem("person_face_sketches", split, sketch_dir, photo_dir))
    return pairs


def fs2k(raw_root):
    pairs = []
    for anno_train in raw_root.rglob("anno_train.json"):
        root = anno_train.parent
        if not (root / "photo").exists() or not (root / "sketch").exists():
            continue

        # FS2K mixes .jpg, .JPG, and .png; annotations omit extensions.
        images = {
            path.relative_to(root).with_suffix("").as_posix().lower(): path
            for folder in (root / "photo", root / "sketch")
            for path in folder.rglob("*")
            if path.suffix.lower() in IMAGE_EXTS
        }

        for filename, split in (("anno_train.json", "train"), ("anno_test.json", "test")):
            anno_path = root / filename
            if not anno_path.exists():
                continue
            entries = json.loads(anno_path.read_text(encoding="utf-8"))
            for item in entries:
                image_name = item["image_name"]
                photo_path = images.get(f"photo/{image_name}".lower())
                sketch_name = image_name.replace("photo", "sketch", 1)
                sketch_name = sketch_name.replace("/image", "/sketch")
                sketch_path = images.get(f"sketch/{sketch_name}".lower())
                if photo_path is not None and sketch_path is not None:
                    pairs.append(row("fs2k", split, sketch_path, photo_path))
    return pairs


def cufs_like(raw_root):
    pairs = []
    for root_name in ("Face_Sketch", "cufs_kaggle"):
        root = raw_root / root_name
        if not root.exists():
            continue

        pairs.extend(
            pair_by_stem(
                root_name.lower(),
                "train",
                root / "sketches",
                root / "photos",
                sketch_suffix="-sz1",
            )
        )
        pairs.extend(
            pair_by_stem(
                root_name.lower(),
                "train",
                root / "cropped_sketch",
                root / "photo",
            )
        )
    return pairs


def natural_key(path):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", path)]


def write_manifest(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        rows,
        key=lambda r: (
            r["dataset"],
            r["split"],
            natural_key(r["sketch_path"]),
            natural_key(r["photo_path"]),
        ),
    )
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["dataset", "split", "sketch_path", "photo_path"]
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--out", default="data/processed/pairs.csv")
    args = parser.parse_args()

    raw_root = Path(args.raw_root).expanduser()
    out_path = Path(args.out).expanduser()

    rows = []
    rows.extend(person_face_sketches(raw_root))
    rows.extend(fs2k(raw_root))
    rows.extend(cufs_like(raw_root))

    write_manifest(rows, out_path)

    counts = {}
    for r in rows:
        counts[(r["dataset"], r["split"])] = counts.get((r["dataset"], r["split"]), 0) + 1
    print(f"wrote {len(rows)} pairs to {out_path}")
    for key, count in sorted(counts.items()):
        print(f"{key[0]:22s} {key[1]:5s} {count:6d}")


if __name__ == "__main__":
    main()
