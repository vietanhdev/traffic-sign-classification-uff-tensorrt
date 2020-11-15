import os
import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications.resnet import ResNet18
from classification_models.tfkeras import Classifiers
import pathlib
import datetime

from tensorflow.keras.backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.log_device_placement = True
sess = tf.Session(config=config)
set_session(sess)

IMG_SIZE = 64
MODEL_DIR = "./models"
TRAIN_DIR = "./dataset/train"
VAL_DIR = "./dataset/val"
LOG_DIR = "./logs"
N_EPOCHS = 100
LEARNING_RATE = 0.0001
NUM_CLASSES = 15

pathlib.Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=15,
    zoom_range=0.1,
    channel_shift_range=0.1)
val_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,  # This is the source directory for training images
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32,
        class_mode='categorical')

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32,
        class_mode='categorical')

ResNet18, preprocess_input = Classifiers.get('resnet18')
base_model = ResNet18(input_shape=(IMG_SIZE,IMG_SIZE,3), weights=None, include_top=False)
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
model = tf.keras.models.Model(inputs=[base_model.input], outputs=[output])

model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

checkpoint_callback = ModelCheckpoint(os.path.join(MODEL_DIR, "model-{epoch:03d}-{val_loss:.3f}.h5"), monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(LOG_DIR, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

class_weight = {
    0: 1, 1: 1.0/100,
    2: 1, 3: 1, 
    4: 1, 5: 1,
    6: 1, 7: 1,
    8: 1, 9: 1,
    10: 1, 11: 1,
    12: 1, 13: 1,
    14: 1
}
model.fit_generator(train_generator, 
    epochs=N_EPOCHS,
    validation_data=validation_generator,
    callbacks=[tensorboard_callback, checkpoint_callback],
    max_queue_size=256,
    workers=12
)