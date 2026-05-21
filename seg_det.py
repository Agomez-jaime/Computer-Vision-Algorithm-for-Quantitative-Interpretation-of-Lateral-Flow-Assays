from segmentation import run_segmentation  # we'll adapt safely below
#from line_detection import detect_lines
from line_detection_color_spaces import detect_lines
import matplotlib.pyplot as plt
from rotation_one_image_at_a_time import align_using_class_23
import random
import os

#WEIGHTS = "Covid LFAs/full_dataset/runs/run_024_b8_e70_optauto_v0.2_flr0.5_fud0.5/weights/best.pt"
#IMAGE   = "Covid LFAs/full_dataset/images/test/1646478897851.jpg"

WEIGHTS = "C:/Users/Asus/PyCharmMiscProject/Covid LFAs/lighting_conditions/random variation/runs/run_002_b16_e100_optauto_v0.2_flr0.5_fud0.5/weights/best.pt"
IMAGE_DIR = "C:/Users/Asus/PyCharmMiscProject/Covid LFAs/lighting_conditions/random variation/images/valid"


image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")]

img_name = random.choice(image_files)
img_path = os.path.join(IMAGE_DIR, img_name)
# ─────────────────────────────────────────────
# STEP 1: get cropped image from segmentation
# ─────────────────────────────────────────────
crop, result = run_segmentation(WEIGHTS, img_path)

crop, _ = align_using_class_23(result, crop)


# ─────────────────────────────────────────────
# STEP 2: run classical detection
# ─────────────────────────────────────────────
'''
num_lines, _, _ = detect_lines(crop)

print("Detected lines:", num_lines)

if num_lines >= 2:
    print("RESULT: Positive (two lines)")
elif num_lines == 1:
    print("RESULT: Negative (one line)")
else:
    print("RESULT: Invalid")
'''
num_lines, best_method, best_kernel, results, signal_map = detect_lines(
    crop,
    methods= ["rgb_bg_corrected"], #["rgb", "hsv_s", "delta_rgb", "rgb_bg_corrected"], #
    kernel_sizes=[61],
    debug=True
)

#plt.imshow(signal_map, cmap='hot')
#plt.colorbar()
#plt.show()

print("Detected lines:", num_lines)