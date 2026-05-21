import os
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np

from segmentation import run_segmentation

WEIGHTS = "C:/Users/Asus/PyCharmMiscProject/Covid LFAs/lighting_conditions/random variation/runs/run_002_b16_e100_optauto_v0.2_flr0.5_fud0.5/weights/best.pt"

IMAGE_DIR = "C:/Users/Asus/PyCharmMiscProject/Covid LFAs/lighting_conditions/random variation/images/valid"

def rotate_and_resize(image, angle):
    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # compute new bounding box size
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # adjust translation so image stays centered
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR)

    return rotated

def draw_fitline(original, pts, vx, vy, x0, y0):
    vis = original.copy()

    h, w = vis.shape[:2]

    # extend line in both directions
    lefty = int((-x0 * vy / vx) + y0)
    righty = int(((w - x0) * vy / vx) + y0)

    pt1 = (0, lefty)
    pt2 = (w - 1, righty)

    # draw fitted line
    cv2.line(vis, pt1, pt2, (0, 255, 0), 2)

    # draw points (class 2/3 centers)
    for p in pts:
        cv2.circle(vis, tuple(p.astype(int)), 4, (0, 0, 255), -1)

    return vis

def align_using_class_23(result, crop, original=None, min_points=2):

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
        print("Not enough class 2/3 points → skipping alignment")
        return crop, None

    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)

    vx = vx[0]
    vy = vy[0]
    x0 = x0[0]
    y0 = y0[0]

    angle = np.degrees(np.arctan2(vy, vx))
    rotate_angle = angle - 90

    h, w = crop.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rotate_angle, 1.0)
    #rotated = cv2.warpAffine(crop, M, (w, h), flags=cv2.INTER_LINEAR)
    rotated = rotate_and_resize(crop, rotate_angle)

    debug_img = None
    if original is not None:
        debug_img = draw_fitline(original, pts, vx, vy, x0, y0)

    return rotated, debug_img

def show_images(original, aligned, title="Alignment test"):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original image")
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Aligned crop")
    plt.imshow(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def test_random_image():
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")]

    img_name = random.choice(image_files)
    img_path = os.path.join(IMAGE_DIR, img_name)

    print(f"Testing: {img_name}")

    crop, result = run_segmentation(WEIGHTS, img_path)

    if crop is None or crop.size == 0:
        print("Segmentation failed")
        return

    original = cv2.imread(img_path)

    aligned, debug = align_using_class_23(result, crop, original=original)

    plt.figure(figsize=(12, 6))

    if debug is not None:
        plt.subplot(1, 3, 1)
        plt.title("FitLine on original")
        plt.imshow(cv2.cvtColor(debug, cv2.COLOR_BGR2RGB))
        plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Crop")
    plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Aligned crop")
    plt.imshow(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.suptitle(img_name)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_random_image()