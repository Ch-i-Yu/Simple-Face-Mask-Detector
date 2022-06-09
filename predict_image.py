# -*- coding: utf-8 -*-

# @Date     : 2022/06/01
# @Author   : Ch'i YU



"""
Face-Mask Detector with OpenCV and Keras

This goal of this project is to roughly identify
wheather a person is wearing a mask or not.

This project aims to:
- Train a model on images of people wearing masks
- Deploy the trained model to faces-masks in images and video streams

This python program aims to:
- Predict a image with pre-trained model

"""



# import required dependencies and libraries
from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils import img_to_array
from keras.models import load_model

import numpy as np
import argparse
import cv2
import os
from natsort import natsorted, ns


# construct parsed arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    "-i", "--image",
    required = True,
    type = str,
    help = "path to input image"
)
ap.add_argument(
    "-d", "--destination",
    type = str,
    default = "./Processed",
    help = "path(directory) to output image"
)
ap.add_argument(
    "-f", "--face",
    type = str,
    default = "./Model/FaceNet",
    help = "path(directory) to face detector"
)
ap.add_argument(
    "-m", "--mask",
    type = str,
    default = "./Model/MaskNet/mask_net.model",
    help = "path to trained mask detector"
)
ap.add_argument(
    "-c", "--confidence",
    type = float,
    default = 0.50,
    help = "minimum probability to filter weak face detections"
)
args = vars(ap.parse_args())



# block base_loging of INFO/WARNING/ERROR
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'



# load pre-trained face detector network
print("<Loading Pre-Trained Face Detector Network......>")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)



# load face-mask detector model
print("<Loading Face-Mask Detector Model......>")
maskNet = load_model(args["mask"])



# load input image with heights & weights
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]        # h => heights; w => weights

# construct a blob of image
blob = cv2.dnn.blobFromImage(
    image,
    1.0,
    (300, 300),
    (104.0, 177.0, 123.0))



# obtain face detections through blob & face detector network
print("<Process Face Detections......>")
faceNet.setInput(blob)
face_detections = faceNet.forward()

# loop over face detections
for i in range(0, face_detections.shape[2]):
    # extract confidence of face detection
    face_confidence = face_detections[0, 0, i, 2]
    
    if face_confidence > args["confidence"]:
        print("<Process Face-Mask Detections......>")

        # compute the (x, y) coordinates of the bounding box of detected face
        face_box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x_start, y_start, x_end, y_end) = face_box.astype("int")

        # adjust the bounding box to fall within dimensions of the frame(image)
        (x_start, y_start) = (max(0, x_start), max(0, y_start))
        (x_end, y_end) = (max(0, x_end), max(0, y_end))

        # extract face ROI
        face = image[y_start:y_end, x_start:x_end]
        
        # convert channel BGR -> RGB
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # resize to fit model
        face = cv2.resize(face, (224, 224))

        # preprocess
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        # predict on face with loaded Face-Mask Detector model
        (with_mask, without_mask) = maskNet.predict(face)[0]

        # determine the label
        if with_mask > without_mask:
            label = "with Mask"     # with_mask
        else:
            label = "without Mask"  # without_mask

        # determine the color of bounding box & bounding description
        if label == "with Mask":
            color = (0, 250, 0)     # green
        else:
            color = (0, 0, 250)     # red

        # add probability to the label
        label = "{}: {:.2f}%".format(label, max(with_mask, without_mask) * 100)

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
fileName = os.path.sep.join([args["destination"], "processed_" + args["image"].split("\\")[-1]])
retval = cv2.imwrite(fileName, image)

if retval:
    print("<processed_" + args["image"].split("\\")[-1] + " saved successfully.>")
else:
    print("<processed_" + args["image"].split("\\")[-1] + " saved unsuccessfully.>")

print("<End of Face-Mask Detection.>")