import os
import shutil
from pathlib import Path

#Move images corresponding to non-empty label files from source to destination

LABELS_DIR = "C:/Users/Asus/PyCharmMiscProject/Covid LFAs/positive_val_test/labels/train" #empty and non-empty .txt 
IMAGES_SRC  = "C:/Users/Asus/PyCharmMiscProject/Covid LFAs/archive/MajorProject/Positive/val_test"  #folder containing the images
IMAGES_DST  = "C:/Users/Asus/PyCharmMiscProject/Covid LFAs/full_dataset/images/positives"

def migrate_images(labels_dir, images_src, images_dst):
    os.makedirs(images_dst, exist_ok=True)
    moved = 0
    skipped = 0

    for txt_file in Path(LABELS_DIR).glob("*.txt"):
        if txt_file.stat().st_size > 0:
            img_file = Path(IMAGES_SRC) / (txt_file.stem + ".jpg")
            if img_file.exists():
                shutil.move(str(img_file), Path(IMAGES_DST) / img_file.name)
                print(f"Moved: {img_file.name}")
                moved += 1
            else:
                print(f"Image not found: {img_file.name}")
        else:
            skipped += 1

    print(f"\nDone. Moved: {moved} | Skipped (empty labels): {skipped}")

if __name__ == "__main__":
    migrate_images(LABELS_DIR, IMAGES_SRC, IMAGES_DST)