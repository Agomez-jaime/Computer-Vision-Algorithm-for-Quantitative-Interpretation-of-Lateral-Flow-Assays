from segmentation import run_segmentation
from line_detection_color_spaces import detect_lines

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os


# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

WEIGHTS = "C:/Users/Asus/PyCharmMiscProject/Covid LFAs/lighting_conditions/random variation/runs/run_002_b16_e100_optauto_v0.2_flr0.5_fud0.5/weights/best.pt"

IMAGE_DIR = "C:/Users/Asus/PyCharmMiscProject/Covid LFAs/lighting_conditions/random variation/images/valid/1649325343789_T43.40570815340037_L-46.59118153614962_G0.7599152428643019.jpg"


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
# REMOVE INVALID REGIONS
# ─────────────────────────────────────────────
'''
def crop_valid_region(image, mask, threshold_ratio=0.99):
    """
    Keep only rows/cols that are almost fully valid.
    Removes black triangles after rotation.
    """

    mask = mask.astype(np.uint8)

    row_valid = np.mean(mask, axis=1) > threshold_ratio
    col_valid = np.mean(mask, axis=0) > threshold_ratio

    ys = np.where(row_valid)[0]
    xs = np.where(col_valid)[0]

    if len(xs) == 0 or len(ys) == 0:
        print("Could not find valid region")
        return image

    y1, y2 = ys[0], ys[-1]
    x1, x2 = xs[0], xs[-1]

    cropped = image[y1:y2 + 1, x1:x2 + 1]

    return cropped
'''

def crop_valid_region(image, mask, padding=10):
    """
    Tight crop around rotated mask.
    Removes all black regions.
    """

    mask = mask.astype(np.uint8)

    coords = cv2.findNonZero(mask)

    if coords is None:
        print("No mask found")
        return image

    x, y, w, h = cv2.boundingRect(coords)

    # optional padding
    x = max(0, x + padding)
    y = max(0, y + padding)

    w = max(1, w - 2 * padding)
    h = max(1, h - 2 * padding)

    cropped = image[y:y+h, x:x+w]

    return cropped

# ─────────────────────────────────────────────
# ALIGNMENT USING CLASS 2/3
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

    print("Rotation angle:", rotate_angle)

    # rotate image
    rotated_img = rotate_and_resize(image, rotate_angle)

    # rotate mask
    mask_uint8 = (mask.astype(np.uint8) * 255)

    rotated_mask = rotate_and_resize(mask_uint8, rotate_angle)

    rotated_mask = rotated_mask > 127

    # crop ONLY valid interior region
    final_crop = crop_largest_inner_rectangle(
    rotated_img,
    rotated_mask
    )

    return final_crop

def largest_inner_rectangle(mask):
    """
    Find largest axis-aligned rectangle fully inside mask.
    Returns x, y, w, h
    """

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
# RANDOM IMAGE
# ─────────────────────────────────────────────


if os.path.isfile(IMAGE_DIR):

    img_path = IMAGE_DIR
    img_name = os.path.basename(img_path)

elif os.path.isdir(IMAGE_DIR):

    image_files = [
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if len(image_files) == 0:
        raise ValueError("No images found in folder")

    img_name = random.choice(image_files)
    img_path = os.path.join(IMAGE_DIR, img_name)

else:
    raise ValueError("IMAGE_DIR is neither a valid file nor folder")

print("Testing:", img_name)


# ─────────────────────────────────────────────
# STEP 1: SEGMENTATION
# ─────────────────────────────────────────────

orig, result, mask = run_segmentation(
    WEIGHTS,
    img_path,
    debug=False
)

# apply mask BEFORE rotation
masked = orig.copy()
masked[~mask] = 0


# ─────────────────────────────────────────────
# STEP 2: ALIGNMENT
# ─────────────────────────────────────────────

crop = align_using_class_23(
    result,
    masked,
    mask
)


# ─────────────────────────────────────────────
# VISUALIZE FINAL CROP
# ─────────────────────────────────────────────
'''
plt.figure(figsize=(8, 8))

plt.title("Final Clean Crop")

plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

plt.axis("off")

plt.show()
'''

# ─────────────────────────────────────────────
# STEP 3: PEAK DETECTION
# ─────────────────────────────────────────────

'''
print("FINAL CROP SHAPE:", crop.shape)

plt.figure(figsize=(5,10))
plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
plt.title("IMAGE SENT TO detect_lines")
plt.axis("off")
plt.show()
'''

num_lines, best_method, best_kernel, results, signal_map = detect_lines(
    crop,
    methods=["delta_rgb"], #["rgb", "hsv_s", "delta_rgb", "rgb_bg_corrected"], 
    kernel_sizes=[17],
    debug=True
)

print("Detected lines:", num_lines)
"""
Tefi estuvo aqui jiji ola t amo MUAK
"""