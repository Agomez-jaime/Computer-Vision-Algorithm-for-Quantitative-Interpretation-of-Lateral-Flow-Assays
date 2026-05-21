from pathlib import Path

src_labels = Path("C:/Users/Asus/PyCharmMiscProject/Covid LFAs/full_dataset/labels")          # original labels
dst_labels = Path("C:/Users/Asus/PyCharmMiscProject/Covid LFAs/full_dataset/labels_merged")   # new labels folder

dst_labels.mkdir(parents=True, exist_ok=True)

remap = {
    0: 0,
    1: 0,
    2: 1,
    3: 2,
    4: 3,
}

for txt_file in src_labels.rglob("*.txt"):

    # recreate identical folder structure
    relative_path = txt_file.relative_to(src_labels)
    out_file = dst_labels / relative_path
    out_file.parent.mkdir(parents=True, exist_ok=True)

    new_lines = []

    with open(txt_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            cls = int(parts[0])
            parts[0] = str(remap[cls])
            new_lines.append(" ".join(parts))

    with open(out_file, "w") as f:
        f.write("\n".join(new_lines) + "\n")