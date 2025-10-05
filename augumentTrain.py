import cv2, random, shutil, os
import numpy as np
from pathlib import Path

# -----------------------------
# 1. AUGMENTATION
# -----------------------------
def augment_image(img_path, label_path, out_img_dir, out_label_dir, num_aug=10):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    with open(label_path, 'r') as f:
        labels = f.readlines()
    base = img_path.stem

    for i in range(num_aug):
        offset = random.randint(20, 80)
        src = np.float32([[0,0],[w,0],[w,h],[0,h]])
        dst = np.float32([
            [random.randint(0,offset), random.randint(0,offset)],
            [w-random.randint(0,offset), random.randint(0,offset)],
            [w-random.randint(0,offset), h-random.randint(0,offset)],
            [random.randint(0,offset), h-random.randint(0,offset)]
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, (w,h))
        if random.random() < 0.5:
            warped = cv2.flip(warped, 1)
            flip = True
        else:
            flip = False
        warped = cv2.convertScaleAbs(warped, alpha=random.uniform(0.8,1.2), beta=random.randint(-20,20))
        cv2.imwrite(str(out_img_dir / f"{base}_aug_{i}.jpg"), warped)

        aug_labels = []
        for line in labels:
            parts = line.strip().split()
            if len(parts) != 5: continue
            cls, x, y, bw, bh = parts
            x_px, y_px = float(x)*w, float(y)*h
            x_new, y_new = cv2.perspectiveTransform(np.array([[[x_px, y_px]]], dtype=np.float32), M)[0][0]
            if flip: x_new = w - x_new
            x_norm, y_norm = x_new/w, y_new/h
            if 0 <= x_norm <= 1 and 0 <= y_norm <= 1:
                aug_labels.append(f"{cls} {x_norm} {y_norm} {bw} {bh}\n")
        with open(out_label_dir / f"{base}_aug_{i}.txt", 'w') as f:
            f.writelines(aug_labels)

# -----------------------------
# 2. AUGMENT ENTIRE DATASET
# -----------------------------
def augment_dataset(original_path, augmented_path, num_aug=10):
    out_img_dir = Path(augmented_path)/"images"
    out_label_dir = Path(augmented_path)/"labels"
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    img_dir = Path(original_path)/"images"
    label_dir = Path(original_path)/"labels"

    for img_file in img_dir.glob("*.[jp][pn]g"):
        label_file = label_dir / f"{img_file.stem}.txt"
        if not label_file.exists(): continue
        shutil.copy(img_file, out_img_dir)
        shutil.copy(label_file, out_label_dir)
        augment_image(img_file, label_file, out_img_dir, out_label_dir, num_aug=num_aug)

# -----------------------------
# 3. SPLIT DATA
# -----------------------------
def split_dataset(dataset_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    img_dir = Path(dataset_path)/"images"
    label_dir = Path(dataset_path)/"labels"

    all_imgs = list(img_dir.glob("*.jpg"))
    random.shuffle(all_imgs)

    n = len(all_imgs)
    n_train = int(n*train_ratio)
    n_val = int(n*val_ratio)

    splits = {
        "train": all_imgs[:n_train],
        "val": all_imgs[n_train:n_train+n_val],
        "test": all_imgs[n_train+n_val:]
    }

    for split, files in splits.items():
        img_split_dir = Path(dataset_path)/"split"/split/"images"
        label_split_dir = Path(dataset_path)/"split"/split/"labels"
        os.makedirs(img_split_dir, exist_ok=True)
        os.makedirs(label_split_dir, exist_ok=True)
        for f in files:
            shutil.copy(f, img_split_dir)
            shutil.copy(label_dir/f"{f.stem}.txt", label_split_dir)

# -----------------------------
# 4. CREATE YAML
# -----------------------------
def create_yaml(dataset_path, nc=4, names=None):
    if names is None:
        names = ['Solid','Stripe','Black','Cue']
    yaml_path = Path(dataset_path)/"split/data.yaml"
    yaml_content = f"path: {dataset_path}/split\ntrain: train/images\nval: val/images\ntest: test/images\nnc: {nc}\nnames: {names}\n"
    with open(yaml_path,'w') as f:
        f.write(yaml_content)
    return str(yaml_path)

# -----------------------------
# 5. RUN EVERYTHING
# -----------------------------
if __name__=="__main__":
    ORIGINAL = "train"              # your current folder
    AUGMENTED = "augmented_data"

    augment_dataset(ORIGINAL, AUGMENTED, num_aug=10)
    split_dataset(AUGMENTED, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    yaml_file = create_yaml(AUGMENTED)
    print(f"Dataset ready. Use YAML file: {yaml_file}")


