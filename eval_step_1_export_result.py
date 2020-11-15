import tensorflow as tf
import cv2
import numpy as np
import sys
import os
from imutils import paths
import time

from tensorflow.keras.backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = False  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

traffic_sign_names = os.listdir("./dataset/train")
traffic_sign_names.sort()
print(traffic_sign_names)

MODEL_PATH = "models/model-039-0.068.h5"
TEST_FOLDER = "dataset/test"
OUTPUT_FILE = "test_result.txt"

model = tf.keras.models.load_model(MODEL_PATH)
images = list(paths.list_images(TEST_FOLDER))

total_time = 0
with open(OUTPUT_FILE, "w") as outfile:
    for i, img_path in enumerate(images):
        print("{} / {}".format(i + 1, len(images)))
        
        img = cv2.imread(img_path)

        begin_time = time.time()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64))
        img = img.astype(float)
        img /= 255.0

        net_input = np.array([img])
        pred = model.predict(net_input)
        pred = np.argmax(pred[0])
        total_time += time.time() - begin_time

        outfile.write("{} {}\n".format(
            os.path.basename(img_path),
            pred
        ))

print("Avg. Time: {}".format(total_time / len(images)))