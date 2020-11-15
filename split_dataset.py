import os
import cv2
from pathlib import Path
import random
import shutil

RAW_IMAGE_DIR = "/mnt/DATA/GRADUATION_RESEARCH/Workstation_Sources/TrafficSignClassification/dataset/raw"
OUTPUT_DIR = "/mnt/DATA/GRADUATION_RESEARCH/Workstation_Sources/TrafficSignClassification/dataset"

VAL_RATIO = 0.1
TEST_RATIO = 0.1

classes = list(os.listdir(RAW_IMAGE_DIR))

random.seed(42)

for cl in classes:

    print(cl)

    class_dir = os.path.join(RAW_IMAGE_DIR, cl)
    images = [i for i in os.listdir(class_dir) if i.endswith(".jpg")]
    n_images = len(images)
    val_num = int(VAL_RATIO * n_images)
    test_num = int(TEST_RATIO * n_images)

    val_images = random.sample(images, val_num)
    images = [i for i in images if i not in val_images]

    test_images = random.sample(images, test_num)
    train_images = [i for i in images if i not in test_images]

    for dset in ["train", "test", "val"]:
        Path(os.path.join(OUTPUT_DIR, dset, cl)).mkdir(parents=True, exist_ok=True)

    assert(n_images == len(train_images) + len(test_images) + len(val_images))

    for i in train_images:
        shutil.copy(os.path.join(class_dir, i), \
            os.path.join(OUTPUT_DIR, "train", cl, i))
    for i in test_images:
        shutil.copy(os.path.join(class_dir, i), \
            os.path.join(OUTPUT_DIR, "test", cl, i))
    for i in val_images:
        shutil.copy(os.path.join(class_dir, i), \
            os.path.join(OUTPUT_DIR, "val", cl, i))

    