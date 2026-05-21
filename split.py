import os
import shutil
import random
from pathlib import Path
from migrate_images import migrate_images

# ── Configure paths ───────────────────────────────────────────────
LABELS_SRC  = "C:/Users/Asus/PyCharmMiscProject/Covid LFAs/mine/labels/train"       # source folder with all .txt label files
DATASET_DST = "C:/Users/Asus/PyCharmMiscProject/Covid LFAs/full_dataset2"       # destination dataset root
IMAGES_SRC  = "C:/Users/Asus/PyCharmMiscProject/Covid LFAs/mine/images/train"   # folder containing the .jpg images

# ── Split configuration ───────────────────────────────────────────
USE_THREE_SPLITS = True   # True → train/valid/test | False → valid/test only

# Define splits as EITHER percentages (must sum to 100) OR absolute counts.
# Set the mode you want and leave the other as None.

SPLIT_MODE = "percentage"   # "percentage" or "count"

# Percentage mode (values must sum to 100):
SPLIT_PERCENTAGES = {
    "train": 80,
    "valid": 10,
    "test":  10,
}

# Count mode (set exact number of files per split; remainder goes to train):
SPLIT_COUNTS = {
    "train": None,  # None = "use whatever is left over"
    "valid": 5,
    "test":  5,
}

RANDOM_SEED = 42   # set to None to disable shuffling
# ─────────────────────────────────────────────────────────────────


def compute_split_sizes(total: int, use_three: bool) -> dict:
    splits = ["train", "valid", "test"] if use_three else ["valid", "test"]

    if SPLIT_MODE == "percentage":
        pcts = {s: SPLIT_PERCENTAGES[s] for s in splits}
        total_pct = sum(pcts.values())
        if total_pct != 100:
            raise ValueError(f"Percentages must sum to 100, got {total_pct}")
        sizes = {}
        assigned = 0
        for i, split in enumerate(splits):
            if i == len(splits) - 1:
                sizes[split] = total - assigned
            else:
                sizes[split] = round(total * pcts[split] / 100)
                assigned += sizes[split]

    elif SPLIT_MODE == "count":
        sizes = {}
        assigned = 0
        for split in splits:
            count = SPLIT_COUNTS.get(split)
            if count is None:
                continue
            if count > total - assigned:
                raise ValueError(
                    f"Not enough files for split '{split}': "
                    f"requested {count}, only {total - assigned} left."
                )
            sizes[split] = count
            assigned += count
        remainder_split = "train" if use_three else splits[0]
        sizes[remainder_split] = total - assigned
        sizes = {s: sizes[s] for s in splits}

    else:
        raise ValueError(f"Unknown SPLIT_MODE: '{SPLIT_MODE}'. Use 'percentage' or 'count'.")

    return sizes


def split_dataset():
    labels_src  = Path(LABELS_SRC)
    images_src  = Path(IMAGES_SRC)
    dataset_dst = Path(DATASET_DST)
    splits = ["train", "valid", "test"] if USE_THREE_SPLITS else ["valid", "test"]

    # Collect non-empty label files from LABELS_SRC that haven't been split yet
    # (i.e. not already present in any destination split folder)
    already_split = set()
    for split in splits:
        for f in (dataset_dst / "labels" / split).glob("*.txt") if (dataset_dst / "labels" / split).exists() else []:
            already_split.add(f.name)

    all_labels = [
        f for f in labels_src.glob("*.txt")
        if f.stat().st_size > 0 and f.name not in already_split
    ]
    total = len(all_labels)
    print(f"Found {total} non-empty label files to split.\n")

    # For each split, also migrate images for labels already in dst (e.g. from a previous partial run)
    for split in splits:
        labels_dst = dataset_dst / "labels" / split
        images_dst = dataset_dst / "images" / split
        if labels_dst.exists():
            pending = [
                f for f in labels_dst.glob("*.txt")
                if not (images_dst / (f.stem + ".jpg")).exists()
            ]
            if pending:
                print(f"── {split.upper()} — migrating {len(pending)} missing images from previous run ──")
                migrate_images(
                    labels_dir=str(labels_dst),
                    images_src=str(images_src),
                    images_dst=str(images_dst),
                )
                print()

    if total == 0:
        print("No new labels to split. Done.")
        return

    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
    random.shuffle(all_labels)

    # Compute how many files go into each split
    sizes = compute_split_sizes(total, USE_THREE_SPLITS)
    print("Split sizes:")
    for split, size in sizes.items():
        print(f"  {split:6s}: {size} files ({size/total*100:.1f}%)")
    print()

    # Slice the shuffled list and process each split
    idx = 0
    for split, size in sizes.items():
        files = all_labels[idx: idx + size]
        idx += size

        labels_dst = dataset_dst / "labels" / split
        images_dst = dataset_dst / "images" / split
        labels_dst.mkdir(parents=True, exist_ok=True)

        print(f"── {split.upper()} ({len(files)} files) ──")

        moved_imgs = 0
        missing_imgs = 0
        for lbl in files:
            # 1. Move the image first (while we still know its name from the label)
            img_file = images_src / (lbl.stem + ".jpg")
            if img_file.exists():
                images_dst.mkdir(parents=True, exist_ok=True)
                shutil.move(str(img_file), images_dst / img_file.name)
                moved_imgs += 1
            else:
                print(f"  [WARNING] Image not found: {img_file.name}")
                missing_imgs += 1

            # 2. Then move the label
            shutil.move(str(lbl), labels_dst / lbl.name)

        print(f"  Labels moved: {len(files)} | Images moved: {moved_imgs}" +
              (f" | Missing: {missing_imgs}" if missing_imgs else ""))
        print()

    print("Done.")
    print(f"Dataset written to: {dataset_dst.resolve()}")


if __name__ == "__main__":
    split_dataset()