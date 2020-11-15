import os
import cv2
import json
import numpy as numpy
from pathlib import Path

JSON_DIR = "/mnt/DATA/DATASETS/mtsd_fully_annotated/annotations"
IMAGE_DIR = "/mnt/DATA/DATASETS/mtsd_fully_annotated/images"
OUTPUT_DIR = "./mapillary"

label_count = {}

json_files = [f for f in list(os.listdir(JSON_DIR)) if f.endswith(".json")]
for i, file in enumerate(json_files):
    print(i)

    full_path = os.path.join(JSON_DIR, file)
    with open(full_path, "r") as f:
        data = json.load(f)

    image_path = os.path.join(IMAGE_DIR, file[:-4] + "jpg")
    image = cv2.imread(image_path)

    if image is None:
        print("Error reading image: {}".format(image_path))
        exit(1)

    traffic_signs = data["objects"]
    crop_id = 0
    for sign in traffic_signs:
        label = sign["label"]

        if label not in label_count.keys():
            label_count[label] = 0
            Path(os.path.join(OUTPUT_DIR, label)).mkdir(parents=True, exist_ok=True)
        else:
            label_count[label] += 1

        bbox = sign["bbox"]
        xmin = int(bbox["xmin"])
        ymin = int(bbox["ymin"])
        ymax = int(bbox["ymax"])
        xmax = int(bbox["xmax"])
        crop_img = image[ymin:ymax, xmin:xmax]
        crop_path = os.path.join(os.path.join(OUTPUT_DIR, label, sign["key"] + ".jpg"))
        cv2.imwrite(crop_path, crop_img)


with open("log.txt", "w") as f:
    for key, value in label_count.items():
        f.write('%s:%s\n' % (key, value))

    

    