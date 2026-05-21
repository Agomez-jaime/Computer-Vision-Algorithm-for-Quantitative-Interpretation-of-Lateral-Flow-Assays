from ultralytics import YOLO
from pathlib import Path
from itertools import product
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────
DATASET_DST = "C:/Users/Asus/PyCharmMiscProject/Covid LFAs/full_dataset2"

# ── Define your hyperparameter grid ───────────────────────────────
# Each key maps to a list of values to try. All combinations will be run.
GRID = {
    "batch":     [16],          # batch size, 16
    "epochs":    [100],        # number of epochs
    "optimizer": ["auto"], # optimizer, Adam
    "hsv_v":     [ 0.2],      # brightness: 0.0 = off, 0.2 = up to 20% darker
    "fliplr":    [ 0.5],      # horizontal flip: 0.0 = off, 0.5 = 50% chance
    "flipud":    [ 0.5],      # vertical flip:   0.0 = off, 0.5 = 50% chance
}

# ── Fixed (not searched) ──────────────────────────────────────────
IMGSZ = 640  # lr0 is left to Ultralytics to set automatically per optimizer

# ── Augmentation settings that are always OFF ─────────────────────
AUGMENTATION_FIXED = {
    "hsv_h":      0.0,   # hue shift        → always OFF (no colour changes)
    "hsv_s":      0.0,   # saturation shift → always OFF (no colour changes)
    "mosaic":     0.0,   # mosaic           → always OFF
    "mixup":      0.0,   # mixup            → always OFF
    "degrees":    0.0,   # rotation         → always OFF
    "translate":  0.0,   # translation      → always OFF
    "scale":      0.0,   # scaling          → always OFF
    "shear":      0.0,   # shear            → always OFF
    "perspective":0.0,   # perspective      → always OFF
}
# ─────────────────────────────────────────────────────────────────


def build_yaml(dataset_dst: Path) -> Path:
    label_files = list((dataset_dst / "labels" / "train").glob("*.txt"))
    class_ids = set()
    for f in label_files:
        for line in f.read_text().splitlines():
            if line.strip():
                class_ids.add(int(line.split()[0]))
    num_classes = len(class_ids)
    print(f"Detected {num_classes} class(es): {sorted(class_ids)}\n")

    yaml_path = dataset_dst / "dataset.yaml"
    yaml_path.write_text(f"""path: {dataset_dst.as_posix()}
train: images/train
val:   images/valid
test:  images/test

nc: {num_classes}
names: {[str(i) for i in sorted(class_ids)]}
""")
    return yaml_path


def run_grid_search():
    dataset_dst = Path(DATASET_DST)
    yaml_path   = build_yaml(dataset_dst)
    runs_dir    = dataset_dst / "runs"

    # Build all combinations
    keys   = list(GRID.keys())
    combos = list(product(*GRID.values()))
    print(f"Total configurations to train: {len(combos)}\n")

    results = []

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))

        # Build a short readable run name
        run_name = (
            f"run_{i+1:03d}"
            f"_b{params['batch']}"
            f"_e{params['epochs']}"
            f"_opt{params['optimizer']}"
            f"_v{params['hsv_v']}"
            f"_flr{params['fliplr']}"
            f"_fud{params['flipud']}"
        )

        print(f"\n{'='*60}")
        print(f"Config {i+1}/{len(combos)}: {params}")
        print(f"{'='*60}")

        model = YOLO("yolo11n-seg.pt")

        model.train(
            data=str(yaml_path),
            project=str(runs_dir),
            name=run_name,
            exist_ok=True,
            verbose=False,
            # Hyperparameters from grid
            batch=params["batch"],
            imgsz=IMGSZ,
            epochs=params["epochs"],
            optimizer=params["optimizer"],
            # Augmentation from grid
            hsv_v=params["hsv_v"],
            fliplr=params["fliplr"],
            flipud=params["flipud"],
            # Augmentation always off
            **AUGMENTATION_FIXED,
        )

        # Collect best val metrics from results.csv
        results_csv = runs_dir / run_name / "results.csv"
        if results_csv.exists():
            df = pd.read_csv(results_csv)
            df.columns = df.columns.str.strip()

            map_col   = [c for c in df.columns if "mask/mAP50" in c or "mAP50" in c]
            map95_col = [c for c in df.columns if "mAP50-95" in c]
            best_row  = df.loc[df[map_col[0]].idxmax()] if map_col else df.iloc[-1]

            results.append({
                "run":        run_name,
                "batch":      params["batch"],
                "epochs":     params["epochs"],
                "optimizer":  params["optimizer"],
                "hsv_v":      params["hsv_v"],
                "fliplr":     params["fliplr"],
                "flipud":     params["flipud"],
                "best_epoch": int(best_row.get("epoch", -1)) + 1,
                "mAP50":      round(float(best_row[map_col[0]]), 4) if map_col else None,
                "mAP50-95":   round(float(best_row[map95_col[0]]), 4) if map95_col else None,
            })
        else:
            print(f"  [WARNING] results.csv not found for {run_name}")

    # ── Summary table ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("GRID SEARCH COMPLETE — RESULTS SUMMARY")
    print(f"{'='*60}\n")

    summary_df = pd.DataFrame(results)
    if not summary_df.empty and "mAP50" in summary_df.columns:
        summary_df = summary_df.sort_values("mAP50", ascending=False).reset_index(drop=True)

    print(summary_df.to_string(index=False))

    summary_path = runs_dir / "grid_search_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")

    if not summary_df.empty:
        best = summary_df.iloc[0]
        print(f"\nBest config:")
        print(f"   mAP50      : {best['mAP50']}")
        print(f"   mAP50-95   : {best['mAP50-95']}")
        print(f"   batch      : {best['batch']}")
        print(f"   epochs     : {best['epochs']}")
        print(f"   optimizer  : {best['optimizer']}")
        print(f"   hsv_v      : {best['hsv_v']}")
        print(f"   fliplr     : {best['fliplr']}")
        print(f"   flipud     : {best['flipud']}")
        print(f"   run folder : {runs_dir / best['run']}")


if __name__ == "__main__":
    run_grid_search()