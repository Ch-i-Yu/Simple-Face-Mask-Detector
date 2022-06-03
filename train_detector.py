# -*- coding: utf-8 -*-

# @Date     : 2022/06/01
# @Author   : Ch'i YU



"""
Face-Mask Detector with OpenCV and Keras(TensorFlow)

This goal of this project is to roughly identify
whether a person is wearing a mask or not.

This project aims to:
- Train a model on images of people wearing masks on Google Colab & Google Drive
- Deploy te trained model to faces-masks in images and video streams

This python program aims to:
- Train a model on images of people wearing masks on Google Colab & Google Drive

"""



# import required dependencies and libraries on google-colab venv
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from imutils import paths

import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os



# construct parsed arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    "-d", "--dataset",
    required = True,
    type = str,
	help = "path to input train-dataset")
ap.add_argument(
    "-p", "--plot",
    type = str,
    default = "Face-Mask-Detector/Plots/plot.png",
	help = "path to output loss & accuracy plot")
ap.add_argument(
    "-m", "--model",
    type = str,
	default = "Face-Mask-Detector/Model/mask_detector.model",
	help = "path to output detector model")
args = vars(ap.parse_args())



# initialize hyperparameters
INIT_LR = 1e-4
EPOCHS = 20
BATCH_SIZE = 32



# load images in train-dataset directory
print("<Loading Images......>")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

if not imagePaths:
    sys.exit("<Error occurred in image path!>")

for imagePath in imagePaths:
    # split imagePath with '\' & extract folder name 
    label = imagePath.split(os.path.sep)[-2]       # i.e. with_mask or without_mask

    # load & preprocess input image
    image = load_img(imagePath, target_size = (224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)

    # append data and labels respectively
    data.append(image)
    labels.append(label)

# convert data & labels to Numpy arrays
data = np.array(data, dtype = "float32")
labels = np.array(labels)



# one-hot encoding convertion
lbs = LabelBinarizer()
labels = lbs.fit_transform(labels)
labels = to_categorical(labels)



# train-test split
(x_train, x_test, y_train, y_test) = train_test_split(
    train_data = data,
    train_target = labels,
    test_size = 0.20,
    stratify= labels,
    random_state = 42,
    shuffle = True
)



# construct training image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range = 20,
	zoom_range = 0.15,
	width_shift_range = 0.2,
	height_shift_range = 0.2,
	shear_range = 0.15,
	horizontal_flip = True,
	fill_mode = "nearest")



# load MobileNetV2 network with outputlayers banned
baseModel = MobileNetV2(
    weights = "imagenet",       # load pre-trained weights
    include_top = False,        # not retain top 3 fully-connected networks
    input_tensor = Input(shape = (224, 224, 3)),
    input_shape = (224, 224, 3) # retain Keras tensor of shape (224, 224, 3)
)

# construct & connect head models
headModel = baseModel.output                                # base model
headModel = AveragePooling2D(pool_size = (7, 7))(headModel) # average pool layer(2d)
headModel = Flatten(name = "flatten")(headModel)            # full-connect layer 
headModel = Dense(128, activation = "relu")(headModel)      # hidden layer
headModel = Dropout(0.5)(headModel)                         # dropout layer
headModel = Dense(2, activation = "softmax")(headModel)     # output layer

# model integration
model = Model(inputs = baseModel.input, outputs = headModel)

# freeze network layers of integrated model
for layer in baseModel.layers:
    layer.trainable = False



# compile model
print("<Compiling Model......>")
optAdam = Adam(lr = INIT_LR, decay = INIT_LR / EPOCHS)
model.compile(
    loss = "binary_crosentropy",
    optimizer = optAdam,
    metrics = ["accuracy"]
)



# train model
print("<Training Model......>")
H = model.fit(
    aug.flow(x_train, y_train, batch_size = BATCH_SIZE),
    steps_per_epoch = len(x_train) // BATCH_SIZE,
    validation_data = (x_test, y_test),
    validation_steps = len(x_test) // BATCH_SIZE,
    epochs = EPOCHS
)


# make predictions on test set
print("<Evaluating Model......>")
predictIndex = model.predict(x_test, batch_size = BATCH_SIZE)
predictIndex = np.argmax(predictIndex, axis = 1)
print(classification_report(
    y_test.argmax(axis = 1),
    predictIndex,
    target_names=lbs.classes_))



# save model
print("<Saving Face-Mask Detector Model......>")
model.save(args["model"], save_format="h5")



# plot training loss & accuracy
plt.style.use("ggplot")
plt.figure()

plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_acc")

plt.title("Training Loss & Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")

print("<Saving Training Loss & Accuracy......>")
plt.savefig(args["plot"])

print("<Successfully Trained Detector Model.>")