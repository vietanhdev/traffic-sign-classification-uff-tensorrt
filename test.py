import tensorflow as tf
import cv2
import numpy as np
import sys
import os

traffic_sign_names = os.listdir("./dataset/train")
traffic_sign_names.sort()
print(traffic_sign_names)

MODEL_PATH = "/mnt/DATA/GRADUATION_RESEARCH/Workstation_Sources/TrafficSignClassification/models/model-039-0.068.h5"
IMAGE = sys.argv[1]

model = tf.keras.models.load_model(MODEL_PATH)

img = cv2.imread(IMAGE)
img = cv2.resize(img, (32, 32))
img = img.astype(float)
img /= 255.0

net_input = np.array([img])
pred = model.predict(net_input)

pred = np.argmax(pred[0])

print(pred)

print(traffic_sign_names[pred])