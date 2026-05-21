import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO


def run_segmentation(WEIGHTS, IMAGE, TARGET_CLASS=0, SHRINK_RATIO=0.05, debug=False):
    # ─────────────────────────────────────────────
    # LOAD MODEL
    # ─────────────────────────────────────────────
    model = YOLO(WEIGHTS)
    result = model.predict(IMAGE, save=False, verbose=False)[0]

    orig = result.orig_img.copy()
    h, w = orig.shape[:2]

    # ─────────────────────────────────────────────
    # EXTRACT MASK
    # ─────────────────────────────────────────────
    class_ids   = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()
    masks       = result.masks.data.cpu().numpy()

    indices = [i for i, c in enumerate(class_ids) if c == TARGET_CLASS]
    if len(indices) == 0:
        raise ValueError("Target class not found")

    best_i = max(indices, key=lambda i: confidences[i])
    mask = masks[best_i]

    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_bool = mask_resized > 0.5

    # ─────────────────────────────────────────────
    # BOUNDING BOX FROM MASK
    # ─────────────────────────────────────────────
    ys, xs = np.where(mask_bool)

    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    # ─────────────────────────────────────────────
    # SHRINK BOX INWARD
    # ─────────────────────────────────────────────
    box_h = y2 - y1
    box_w = x2 - x1

    dy = int(box_h * SHRINK_RATIO)
    dx = int(box_w * SHRINK_RATIO)

    y1_new = max(0, y1 + dy)
    y2_new = min(h, y2 - dy)
    x1_new = max(0, x1 + dx)
    x2_new = min(w, x2 - dx)

    crop = orig[y1_new:y2_new, x1_new:x2_new].copy()

    # ─────────────────────────────────────────────
    # VISUALIZATION (UNCHANGED)
    # ─────────────────────────────────────────────
    orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    overlay = orig_rgb.copy()
    overlay[mask_bool] = [0, 255, 0]

    vis = orig_rgb.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.rectangle(vis, (x1_new, y1_new), (x2_new, y2_new), (0, 255, 0), 2)

    yolo_vis = result.plot()
    yolo_vis = cv2.cvtColor(yolo_vis, cv2.COLOR_BGR2RGB)

    if debug:
        plt.figure(figsize=(16, 5))

        plt.subplot(1, 4, 1)
        plt.title("Original")
        plt.imshow(orig_rgb)
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.title("YOLO Prediction")
        plt.imshow(yolo_vis)
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.title("Mask + Boxes")
        plt.imshow(vis)
        plt.axis("off")

        plt.subplot(1, 4, 4)
        plt.title("Final Crop (Clean)")
        plt.imshow(crop_rgb)
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    
    return orig, result, mask_bool

'''
# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
WEIGHTS = "Covid LFAs/full_dataset/runs/run_024_b8_e70_optauto_v0.2_flr0.5_fud0.5/weights/best.pt"
IMAGE   = "Covid LFAs/full_dataset/images/test/1646478897851.jpg"
TARGET_CLASS = 0

# how much to shrink the box (percentage of size)
SHRINK_RATIO = 0.05   # 5% inward (tune this!)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
model = YOLO(WEIGHTS)
result = model.predict(IMAGE, save=False, verbose=False)[0]

orig = result.orig_img.copy()
h, w = orig.shape[:2]

# ─────────────────────────────────────────────
# EXTRACT MASK
# ─────────────────────────────────────────────
class_ids   = result.boxes.cls.cpu().numpy().astype(int)
confidences = result.boxes.conf.cpu().numpy()
masks       = result.masks.data.cpu().numpy()

indices = [i for i, c in enumerate(class_ids) if c == TARGET_CLASS]
if len(indices) == 0:
    raise ValueError("Target class not found")

best_i = max(indices, key=lambda i: confidences[i])
mask = masks[best_i]

mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
mask_bool = mask_resized > 0.5

# ─────────────────────────────────────────────
# BOUNDING BOX FROM MASK
# ─────────────────────────────────────────────
ys, xs = np.where(mask_bool)

y1, y2 = ys.min(), ys.max()
x1, x2 = xs.min(), xs.max()

# ─────────────────────────────────────────────
# SHRINK BOX INWARD (KEY STEP)
# ─────────────────────────────────────────────
box_h = y2 - y1
box_w = x2 - x1

dy = int(box_h * SHRINK_RATIO)
dx = int(box_w * SHRINK_RATIO)

y1_new = max(0, y1 + dy)
y2_new = min(h, y2 - dy)
x1_new = max(0, x1 + dx)
x2_new = min(w, x2 - dx)

crop = orig[y1_new:y2_new, x1_new:x2_new].copy()

# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────
orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

# overlay mask
overlay = orig_rgb.copy()
overlay[mask_bool] = [0, 255, 0]

# draw original bbox (red) and shrunk bbox (green)
vis = orig_rgb.copy()
cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)      # original (red)
cv2.rectangle(vis, (x1_new, y1_new), (x2_new, y2_new), (0, 255, 0), 2)  # shrunk (green)

# YOLO vis
yolo_vis = result.plot()
yolo_vis = cv2.cvtColor(yolo_vis, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(16, 5))

plt.subplot(1, 4, 1)
plt.title("Original")
plt.imshow(orig_rgb)
plt.axis("off")

plt.subplot(1, 4, 2)
plt.title("YOLO Prediction")
plt.imshow(yolo_vis)
plt.axis("off")

plt.subplot(1, 4, 3)
plt.title("Mask + Boxes")
plt.imshow(vis)
plt.axis("off")

plt.subplot(1, 4, 4)
plt.title("Final Crop (Clean)")
plt.imshow(crop_rgb)
plt.axis("off")

plt.tight_layout()
plt.show()
'''