# peak_intensity_values_all2.py

import os
import cv2
import csv
import numpy as np
import glob
from segmentation import run_segmentation
from line_detection_color_spaces2 import detect_lines


# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

WEIGHTS = "C:/Users/Asus/PyCharmMiscProject/Covid LFAs/lighting_conditions/random variation/runs/run_002_b16_e100_optauto_v0.2_flr0.5_fud0.5/weights/best.pt"

IMAGE_DIR = "C:/Users/Asus/PyCharmMiscProject/Covid LFAs/lighting_conditions/random variation/images/valid"

LABEL_DIR = "C:/Users/Asus/PyCharmMiscProject/Covid LFAs/lighting_conditions/random variation/labels/valid"

OUTPUT_CSV = "C:/Users/Asus/PyCharmMiscProject/Covid LFAs/lighting_conditions/random variation/results_valid_delta_rgb.csv"

ARCHIVE_ROOT = r"C:\Users\Asus\PyCharmMiscProject\Covid LFAs\archive\MajorProject"


# ─────────────────────────────────────────────
# ROTATION
# ─────────────────────────────────────────────

def rotate_and_resize(image, angle):

    h, w = image.shape[:2]

    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(
        image,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return rotated


# ─────────────────────────────────────────────
# MAXIMUM INNER RECTANGLE
# ─────────────────────────────────────────────

def largest_inner_rectangle(mask):

    mask = mask.astype(np.uint8)

    h, w = mask.shape

    heights = np.zeros(w, dtype=int)

    best_area = 0
    best_rect = (0, 0, w, h)

    for y in range(h):

        for x in range(w):

            if mask[y, x]:
                heights[x] += 1
            else:
                heights[x] = 0

        stack = []

        x = 0

        while x <= w:

            curr_height = heights[x] if x < w else 0

            if not stack or curr_height >= heights[stack[-1]]:
                stack.append(x)
                x += 1

            else:

                top = stack.pop()

                height = heights[top]

                width = x if not stack else x - stack[-1] - 1

                area = height * width

                if area > best_area:

                    best_area = area

                    rect_x = 0 if not stack else stack[-1] + 1
                    rect_y = y - height + 1

                    best_rect = (
                        rect_x,
                        rect_y,
                        width,
                        height
                    )

    return best_rect


def crop_largest_inner_rectangle(image, mask):

    x, y, w, h = largest_inner_rectangle(mask)

    cropped = image[y:y+h, x:x+w]

    return cropped


# ─────────────────────────────────────────────
# ALIGNMENT
# ─────────────────────────────────────────────

def align_using_class_23(result, image, mask, min_points=2):

    boxes = result.boxes

    cls = boxes.cls.cpu().numpy().astype(int)
    xyxy = boxes.xyxy.cpu().numpy()

    pts = []

    for i, c in enumerate(cls):

        if c in [2, 3]:

            x1, y1, x2, y2 = xyxy[i]

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            pts.append([cx, cy])

    pts = np.array(pts, dtype=np.float32)

    if len(pts) < min_points:

        print("Not enough class 2/3 points")

        return image

    vx, vy, x0, y0 = cv2.fitLine(
        pts,
        cv2.DIST_L2,
        0,
        0.01,
        0.01
    )

    vx = vx[0]
    vy = vy[0]

    angle = np.degrees(np.arctan2(vy, vx))

    rotate_angle = angle - 90

    rotated_img = rotate_and_resize(
        image,
        rotate_angle
    )

    h, w = mask.shape[:2]

    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(
        center,
        rotate_angle,
        1.0
    )

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    mask_uint8 = (mask.astype(np.uint8) * 255)

    rotated_mask = cv2.warpAffine(
        mask_uint8,
        M,
        (new_w, new_h),
        flags=cv2.INTER_NEAREST
    )

    rotated_mask = rotated_mask > 127

    final_crop = crop_largest_inner_rectangle(
        rotated_img,
        rotated_mask
    )

    return final_crop


# ─────────────────────────────────────────────
# GROUND TRUTH
# ─────────────────────────────────────────────


def get_ground_truth(img_name):

    # image id = first token before "_"
    image_id = img_name.split("_")[0]

    # recursively search inside positive folder
    positive_matches = glob.glob(
        os.path.join(
            ARCHIVE_ROOT,
            "positive",
            "**",
            f"{image_id}*"
        ),
        recursive=True
    )

    if len(positive_matches) > 0:
        return 1

    # recursively search inside negative folder
    negative_matches = glob.glob(
        os.path.join(
            ARCHIVE_ROOT,
            "negative",
            "**",
            f"{image_id}*"
        ),
        recursive=True
    )

    if len(negative_matches) > 0:
        return 0

    print(f"GT NOT FOUND: {img_name}")

    return None

'''
def get_ground_truth(label_path):

    if not os.path.exists(label_path):
        return None

    with open(label_path, "r") as f:
        lines = f.readlines()

    classes = [
        int(l.split()[0])
        for l in lines
        if len(l.strip()) > 0
    ]

    if len(classes) == 0:
        return None

    return 0 if 0 in classes else None
'''

# ─────────────────────────────────────────────
# PEAK EXTRACTION
# ─────────────────────────────────────────────

def extract_peak_intensities(smoothed, peaks):

    return [
        float(smoothed[p])
        for p in peaks
    ]


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

results = []

image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.endswith(".jpg")
]

print(image_files)

for img_name in image_files:

    img_path = os.path.join(
        IMAGE_DIR,
        img_name
    )

    label_path = os.path.join(
        LABEL_DIR,
        img_name.replace(".jpg", ".txt")
    )

    gt = get_ground_truth(img_name)

    if gt is None:
        continue

    try:

        # ─────────────────────────────────────
        # SEGMENTATION
        # ─────────────────────────────────────

        orig, result, mask = run_segmentation(
            WEIGHTS,
            img_path
        )

        # ─────────────────────────────────────
        # ALIGNMENT
        # ─────────────────────────────────────

        crop = align_using_class_23(
            result,
            orig,
            mask
        )

        if crop is None or crop.size == 0:

            print("EMPTY CROP")

            continue

        # ─────────────────────────────────────
        # DETECTION
        # ─────────────────────────────────────

        num_lines, best_method, best_kernel, all_results, signal_map = detect_lines(
            crop,
            methods=[
                "delta_rgb"
            ],#"rgb","delta_rgb", "rgb_bg_corrected"
            kernel_sizes=[15],
            debug=False
        )

        # ─────────────────────────────────────
        # BEST RESULT
        # ─────────────────────────────────────

        best = all_results[best_method][best_kernel]

        peaks = best["peaks"]

        smoothed = best["smoothed"]

        peak_intensities = extract_peak_intensities(
            smoothed,
            peaks
        )

        # ─────────────────────────────────────
        # PREDICTION
        # ─────────────────────────────────────

        if num_lines >= 2:
            pred = 1

        elif num_lines == 1:
            pred = 0

        else:
            pred = -1

        # ─────────────────────────────────────
        # SAVE RESULT
        # ─────────────────────────────────────

        results.append([
            img_name,
            gt,
            pred,
            num_lines,
            peak_intensities
        ])

        print(f"Processed {img_name}")

    except Exception as e:

        print(f"Error on {img_name}: {e}")


# ─────────────────────────────────────────────
# SAVE CSV
# ─────────────────────────────────────────────

with open(OUTPUT_CSV, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "image_name",
        "ground_truth",
        "prediction",
        "num_peaks",
        "peak_intensities"
    ])

    writer.writerows(results)

print(f"\nSaved results to {OUTPUT_CSV}")


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

TP = FP = TN = FN = 0

for row in results:

    _, gt, pred, _, _ = row

    gt_pos = (gt == 1)
    pred_pos = (pred == 1)

    if gt_pos and pred_pos:
        TP += 1

    elif not gt_pos and pred_pos:
        FP += 1

    elif not gt_pos and not pred_pos:
        TN += 1

    elif gt_pos and not pred_pos:
        FN += 1


precision = TP / (TP + FP + 1e-9)

recall = TP / (TP + FN + 1e-9)

f1 = 2 * precision * recall / (
    precision + recall + 1e-9
)

accuracy = (TP + TN) / (
    TP + TN + FP + FN + 1e-9
)


print("\n──────── METRICS ────────")

print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")

print(f"Precision: {precision:.4f}")

print(f"Recall:    {recall:.4f}")

print(f"F1-score:  {f1:.4f}")

print(f"Accuracy:  {accuracy:.4f}")

print("RESULTS LENGTH:", len(results))