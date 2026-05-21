import os
import shutil

original_labels_dir = "C:\\Users\\Asus\\PyCharmMiscProject\\Covid LFAs\\full_dataset\\labels"
images_dir = "C:\\Users\\Asus\\PyCharmMiscProject\\Covid LFAs\\lighting_conditions\\random variation\\images"
new_labels_dir = "C:\\Users\\Asus\\PyCharmMiscProject\\Covid LFAs\\lighting_conditions\\random variation\\labels"
split = ["train","valid", "test"] #

# Create output folder if it doesn't exist
os.makedirs(new_labels_dir, exist_ok=True)
for s in split:
    label_path = os.path.join(original_labels_dir, s)
    image_path = os.path.join(images_dir, s)
    new_labels_dir_split = os.path.join(new_labels_dir, s)
    for img_file in os.listdir(image_path):
        if not img_file.endswith((".jpg", ".jpeg", ".png")):
            continue

        # Extract original ID (before first underscore)
        original_id = img_file.split("_")[0]

        src_label = os.path.join(label_path, original_id + ".txt")
        dst_label = os.path.join(new_labels_dir_split, os.path.splitext(img_file)[0] + ".txt")

        if os.path.exists(src_label):
            shutil.copy(src_label, dst_label)
        else:
            print(f"⚠️ Missing label for: {img_file}")
    print("Done copying labels.")
