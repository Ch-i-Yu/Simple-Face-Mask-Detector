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
- Predict a video with pre-trained model

"""


# import required dependencies and libraries
from keras.applications.mobilenet_v2 import preprocess_input
from keras_preprocessing.image import img_to_array
from keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os



# construct parsed arguments
ap = argparse.ArgumentParser()
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
ap.add_argument(
    "-v", "--video",
    type = bool,
    default = False,
    help = "save captured video"
)
args = vars(ap.parse_args())



# block base_loging of INFO/WARNING/ERROR
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

img_array = []


def detect_and_predict_face_mask(frame, faceNet, maskNet):
    """
    Scan through a single video frame to detect faces with masks.

    Args:
        frame:
            A single video frame(image).

        faceNet:
            A pre-trained caffe-based model to detect faces in images.

        maskNet:
            A pre-trained h5 model to detect faces with masks.

    Returns:
        A 2-tuple of:
            locs: face locations
            preds: predictions corresponding to the face locations

    Raises:
        None
    
    """

    # load input image with heights & weights
    (h, w) = frame.shape[:2]

    # construct a blob of image
    blob = cv2.dnn.blobFromImage(
    frame,
    1.0,
    (300, 300),
    (104.0, 177.0, 123.0))

    # initialize lists of faces / corresponding locations / face-mask predictions
    faces = []
    locs = []
    preds = []

    # obtain face detections through blob & face detector network
    faceNet.setInput(blob)
    face_detections = faceNet.forward()

    # loop over face detections
    for i in range(0, face_detections.shape[2]):
        # extract confidence of face detection
        face_confidence = face_detections[0, 0, i, 2]

        if face_confidence > args["confidence"]:
        # compute the (x, y) coordinates of the bounding box of detected face
            face_box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x_start, y_start, x_end, y_end) = face_box.astype("int")

            # adjust the bounding box to fall within dimensions of the frame(image)
            (x_start, y_start) = (max(0, x_start), max(0, y_start))
            (x_end, y_end) = (max(0, x_end), max(0, y_end))

            # extract face ROI
            face = frame[y_start:y_end, x_start:x_end]

            # convert channel BGR -> RGB
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            # resize to fit model
            face = cv2.resize(face, (224, 224))

            # preprocess
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and corresponding boxs to their respective lists
            faces.append(face)
            locs.append((x_start, y_start, x_end, y_end))

        # make predictions if at least one face was detected
    if len(faces) > 0:
        faces = np.array(faces, dtype = "float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)



# load pre-trained face detector network
print("<Loading Pre-Trained Face Detector Network......>")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)



# load face-mask detector model
print("<Loading Face-Mask Detector Model......>")
maskNet = load_model(args["mask"])



# initialize the video stream on a webcam
print("<Starting video stream......>")
video = VideoStream(src=0).start()
time.sleep(2.0)         # a gentle pause to allow the camera to warm up

print("<NOTE: Press `ESC` to quit detections!>")



# loop over the frames from the video stream
while True:
    # load a frame from the threaded video stream
    frame = video.read()

    # resize frame
    frame = imutils.resize(frame, width = 600)

    # detect faces-masks in the frame
    (locs, preds) = detect_and_predict_face_mask(frame, faceNet, maskNet)

    # loop over face detections
    for (loc, pred) in zip(locs, preds):
        # unpack the locations(bounding box) and predictions
        (x_start, y_start, x_end, y_end) = loc
        (with_mask, without_mask) = pred

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
            img = frame,
            text = label,
            org = (x_start, y_start-10),
            fontFace = cv2.FONT_HERSHEY_SIMPLEX,
            fontScale = 0.8,
            color = color,
            thickness = 2
        )   #label
        cv2.rectangle(
            img = frame, 
            pt1 = (x_start, y_start),
            pt2 = (x_end, y_end),
            color = color, 
            thickness = 3
        )   # bounding box
    
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(10)

    img_array.append(frame)
    height, width, layers = frame.shape
    size = (width, height)
    
    # if the ESC key is pressed, break from loop
    if key == 27:       # ESC
        print("<Ending video stream......>")

        # clean up
        cv2.destroyAllWindows()
        video.stop()
        break
video.stop()
cv2.destroyAllWindows()



# Save all output frames as MP4 video
if args["video"]:
    print("<Saving video......>")
    output_path = os.path.sep.join(["./Processed", time.asctime().replace(' ', '_').replace(':', '-') + ".mp4"])
    output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 12, size)

    for i in range(len(img_array)):
        output.write(img_array[i])
    output.release()
    print("<Successfully saved video.>")

print("<End of Face-Mask Detection.>")