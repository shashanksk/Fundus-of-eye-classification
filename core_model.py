from __future__ import print_function
import csv
import math
import sys
import cv2
import os
from sklearn.metrics import f1_score
from pathlib import Path
from keras import optimizers
import numpy as np
from keras.models import load_model
import time
from unet import *
from data import *
from keras.models import Model
from keras.callbacks import ModelCheckpoint, CSVLogger
import keras.backend as K
K.set_image_data_format('channels_last')


def recall_m(y_true, y_pred):
    y_pred = K.reshape(y_pred, shape=(
        (K.shape(y_pred)[0]*K.shape(y_pred)[1]*K.shape(y_pred)[2]), num_classes))
    y_true = K.reshape(y_true, shape=(
        (K.shape(y_true)[0]*K.shape(y_true)[1]*K.shape(y_true)[2]), num_classes))
    # without background. Last class should be background
    true_positives = K.sum(
        K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=0)[0:-1])
    # without background. Last class should be background
    possible_positives = K.sum(
        K.sum(K.round(K.clip(y_true, 0, 1)), axis=0)[0:-1])
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    y_pred = K.reshape(y_pred, shape=(
        (K.shape(y_pred)[0]*K.shape(y_pred)[1]*K.shape(y_pred)[2]), num_classes))
    y_true = K.reshape(y_true, shape=(
        (K.shape(y_true)[0]*K.shape(y_true)[1]*K.shape(y_true)[2]), num_classes))
    # without background. Last class should be background
    true_positives = K.sum(
        K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=0)[0:-1])
    # without background. Last class should be background
    predicted_positives = K.sum(
        K.sum(K.round(K.clip(y_pred, 0, 1)), axis=0)[0:-1])
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def fscore(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall + K.epsilon()))


def jaccard_distance_loss(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def fscore1(y_true, y_pred):
    y_true = y_true[:, :, :, :-1]
    y_pred = y_pred[:, :, :, :-1]
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    true_positives = K.sum(K.round(y_true * y_pred))
    predicted_positives = K.sum(K.round(y_pred))
    possible_positives = K.sum(K.round(y_true))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f_score1 = 2*((precision*recall)/(precision+recall + K.epsilon()))
    return f_score1


data_gen_train = dict(rotation_range=10,
                      horizontal_flip=True,
                      width_shift_range=5,
                      height_shift_range=5,
                      vertical_flip=True,
                      fill_mode='nearest')

path = os.path.dirname(os.getcwd())
path = Path(path)

print(path)


# path1 = Path("/mnt/smiledata/shajahan_trails/Weights")
path1 = Path("/Weights")


magnification = "4x"
if sys.argv[1:]:
    magnification = sys.argv[1]

# val_name = path/"Data_keras"/magnification/"val"
# train_name = path/"Data_keras"/magnification/"train"
# csv_name = path/"Results/CSV"/magnification/"training_4x.csv"
# weights_folder = path1/magnification/"weights"
# model_name = path/"Modelh5/core_model.h5"

val_name = "data_keras/val"
train_name = "data_keras/train"
csv_name = "Results/CSV/training_4x.csv"
weights_folder = "Weights/4x/weights"
model_name = "Modelh5/core_model.h5"


batch_size = 4  # No. of images in a batch
size = 512
weights_40x = [0.979, 0.947, 0.929, 0.982, 0.164]
weights_10x = [0.967, 0.96, 0.88, 0.974, 0.219]
weights_4x = [0.967, 0.969, 0.887, 0.969, 0.208]
weights = weights_4x

num_images = num_of_images(train_name)

# val_x, val_y, img_name = validation(
#     str(val_name/"images"), str(val_name/"mask")) i commentd

val_x, val_y, img_name = validation(
    "data_keras/val/images", "data_keras/val/mask")

train = dataGenerator(batch_size, train_name, data_gen_train, size)
loss_fn = 'categorical_crossentropy'
batch_steps = np.ceil(num_images / batch_size)
sgd = optimizers.SGD(lr=1e-6, decay=1e-6, momentum=0.9, nesterov=True)
rms = optimizers.RMSprop(lr=1e-4)
#opt = 'adam'
opt = rms

if os.path.isdir(weights_folder) is not True:
    os.mkdir(weights_folder)

csv_logger = CSVLogger(csv_name, append=True)
# ,save_best_only=True,monitor='val_fscore',mode='max')
# checkpointer = ModelCheckpoint(
#     filepath='%s/weights.{epoch:03d}.hdf5' % weights_folder, save_weights_only=True) i commented
start = time.time()


# checkpoint_path = str(model_name)

checkpoint_path = None

if checkpoint_path is not None:
    model = load_model(checkpoint_path, custom_objects={'fscore': fscore})
    # To continue from a particular epoch
    # model.load_weights("%s/weights.120.hdf5" % str(weights_folder)) i commented
    initial_epoch = 120
else:
    model = build_model(input_shape=(None, None, 3),
                        preset_model="MobileUNet-Skip", num_classes=num_classes)
    initial_epoch = 0

final_epoch = 200
print(model.summary())
model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy', fscore])

# model.fit_generator(train, steps_per_epoch=batch_steps, epochs=final_epoch, verbose=1, validation_data=(
#     val_x, val_y), validation_steps=None,  callbacks=[csv_logger, checkpointer], shuffle=True, class_weight=weights, initial_epoch=initial_epoch)
# i commeted
model.fit_generator(train, steps_per_epoch=batch_steps, epochs=final_epoch, verbose=1, validation_data=(
    val_x, val_y), validation_steps=None, shuffle=True, class_weight=weights, initial_epoch=initial_epoch)

model.save(str(model_name))
end = time.time()
print(
    f"time taken for training {final_epoch - initial_epoch} is {end - start} seconds")
