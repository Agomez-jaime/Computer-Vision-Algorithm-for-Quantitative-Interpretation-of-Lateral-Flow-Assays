from ultralytics import YOLO
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────
DATASET_DST = "C:/Users/Asus/PyCharmMiscProject/Covid LFAs/full_dataset"
RUNS_DIR    = "C:/Users/Asus/PyCharmMiscProject/Covid LFAs/full_dataset/runs"

# ── Pick the run you want to test ─────────────────────────────────
RUN_NAME = "run_024_b8_e70_optauto_v0.2_flr0.5_fud0.5"   # ← change this
# ─────────────────────────────────────────────────────────────────


def test_run():
    run_dir   = Path(RUNS_DIR) / RUN_NAME
    weights   = run_dir / "weights" / "best.pt"
    yaml_path = Path(DATASET_DST) / "dataset.yaml"
    save_dir  = run_dir / "test"

    if not weights.exists():
        print(f"[ERROR] No weights found at: {weights}")
        return

    print(f"Testing : {RUN_NAME}")
    print(f"Weights : {weights}")
    print(f"Output  : {save_dir}\n")

    model = YOLO(str(weights))

    metrics = model.val(
        data=str(yaml_path),
        split="test",          # use test split, not val
        save_dir=save_dir,
        plots=True,
        verbose=True,
    )

    # ── Print summary ─────────────────────────────────────────────
    seg = metrics.seg if hasattr(metrics, "seg") else metrics.box
    p      = float(seg.p.mean())
    r      = float(seg.r.mean())
    map50  = float(seg.map50)
    map95  = float(seg.map)
    f1     = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    print(f"\n── Test Results: {RUN_NAME} ──")
    print(f"  mAP50     : {map50:.4f}")
    print(f"  mAP50-95  : {map95:.4f}")
    print(f"  Precision : {p:.4f}")
    print(f"  Recall    : {r:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"\nAll outputs saved to: {save_dir.resolve()}")


if __name__ == "__main__":
    test_run()