import os
import cv2
from pathlib import Path
import random
import shutil

N_SAMPLE = 20000
RAW_IMAGE_DIR = "/mnt/DATA/GRADUATION_RESEARCH/Workstation_Sources/TrafficSignClassification/mapillary_crop"
OUTPUT_DIR = "/mnt/DATA/GRADUATION_RESEARCH/Workstation_Sources/TrafficSignClassification/dataset/raw/other"


def check_keywords(path):
    avoid_keywords = [
        "regulatory--end-of-maximum-speed-limit-70",
        "regulatory--end-of-speed-limit-zone",
        "regulatory--maximum-speed-limit",
        "regulatory--no-overtaking"
    ]
    for word in avoid_keywords:
        if word in path:
            return False
    return True

from glob import glob
images = [y for x in os.walk(RAW_IMAGE_DIR) for y in glob(os.path.join(x[0], '*.jpg'))]
images = [i for i in images if check_keywords(i)]

samples = random.sample(images, N_SAMPLE)
for sample in samples:
    basename = os.path.basename(sample)
    shutil.copy(sample, os.path.join(OUTPUT_DIR, basename))

    