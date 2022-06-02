# -*- coding: utf-8 -*-

# @Date     : 2022/06/01
# @Author   : Ch'i YU



"""
Face-Mask Detector with OpenCV and Keras(TensorFlow)

This goal of this project is to roughly identify
wheather a person is wearing a mask or not.

This project aims to:
- Train a model on images of people wearing masks on Google Colab & Google Drive
- Deploy te trained model to faces-masks in images and video streams

This python program aims to:
- Predict a image with pre-trained model

"""



# import required dependencies and libraries
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.models import load_model

import numpy as np
import argparse
import cv2
import os
from natsort import natsorted, ns


# construct parsed arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    "-s", "--src",
    required = True,
    type = str,
    help = "path to input image"
)
ap.add_argument(
    "-d", "--dst",
    type = str,
    default = "Face-Mask-Detector/Processed",
    help = "path(directory) to output image"
)
ap.add_argument(
    "-px", "--prototxt",
    type = str,
    default = "Face-Mask-Detector/Model/deploy.prototxt",
    help = "path to prototxt of face detector"
)
ap.add_argument(
    "-ws", "--weights",
    type = str,
    default = "Face-Mask-Detector/Model/res10_300x300_ssd_iter_140000.caffemodel",
    help = "path to caffemodel of face detector"
)
ap.add_argument(
    "-pl", "--plot",
    type = str,
    default = "Face-Mask-Detector/Plots/predict.png",
    help = "path to output prediction"
)
args = vars(ap.parse_args())



# load serialized face detector network
print("Loading Serialized Face Detector Network......")
prototxtPath = args["prototxt"]
weightsPath = args["weights"]
net = cv2.dnn.readNet(prototxtPath, weightsPath)



# load face-mask detector model
print("Loading Face-Mask Detector Model......")
model = load_model("mask_detector.model")



# load input image with heights & weights
image = cv2.imread(args["src"])
(h, w) = image.shape[:2]        # h => heights; w => weights

# construct a blob of image
blob = cv2.dnn.blobFromImage(
    image,
    1.0,
    (300, 300),
    (104.0, 177.0, 123.0))



# obtain face detections through blob & face detector network
print("<Process Face Detections......>")
net.setInput(blob)
face_detections = net.forward()

# loop over face detections
for i in range(0, face_detections.shape[2]):
    # extract confidence of face detection
    face_confidence = face_detections[0, 0, i, 2]
    
    if face_confidence > 0.5:
        print("<Process Face-Mask Detections......>")

        # compute the (x, y) coordinates of the bounding box of detected face
        face_box = face_detections[0, 0, i, 7] * np.array([w, h, w, h])
        (x_start, y_start, x_end, y_end) = face_box.astype("int")

        # adjust the bounding box to fall within dimensions of the frame(image)
        (x_start, y_start) = (max(0, x_start), max(0, y_start))
        (x_end, y_end) = (max(0, x_end), max(0, y_end))

        # extract face ROI
        face = image[y_start:y_end, x_start:x_end]
        
        # convert channel BGR -> RGB
        face = cv2.cvtColor(face, cv2.COLORBGR2RGB)

        # resize to fit model
        face = cv2.resize(face, (224, 224))

        # preprocess
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        # predict on face with loaded Face-Mask Detector model
        (withMask, withoutMask) = model.predict(face)[0]

        # determine the label
        if withMask > withoutMask:
            label = "withMask"
        else:
            label = "withoutMask"

        # determine the color of bounding box & bounding description
        if label == "withMask":
            color = (0, 250, 0)     # green
        else:
            color = (0, 0, 250)     # red

        # put label & bounding box on output image
        cv2.putText(
            img = image,
            text = label,
            org = (x_start, y_start-10),
            fontFace = cv2.FONT_HERSHEY_SIMPLEX,
            fontScale = 0.45,
            color = color,
            thickness = 2
        )   #label
        cv2.rectangle(
            img = image, 
            pt1 = (x_start, y_start),
            pt2 = (x_end, y_end),
            color = color, 
            thickness = 2
        )   # bounding box

        # save output image
        fileName = args["dst"] + "processed_" + args["src"].split("/")[-1]
        retval = cv2.imwrite(fileName)

        if retval:
            print("<processed_" + args["src"].split("/")[-1] + " saved successfuly.>")
        else:
            print("<processed_" + args["src"].split("/")[-1] + " saved unsuccessfuly.>")

print("<End of Face-Mask Detection.>")