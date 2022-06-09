# -*- coding: utf-8 -*-

# @Date     : 2022/06/09
# @Author   : Ch'i YU



"""
Face-Mask Detector with OpenCV and Keras

This goal of this project is to roughly identify
whether a person is wearing a mask or not.

This project aims to:
- Train a model on images of people wearing masks
- Deploy the trained model to faces-masks in images and video streams

This python program aims to:
- Provide and deploy the trained model with a user-friendly Streamlit App

"""



# import required dependencies and libraries
from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils import img_to_array
from keras.models import load_model

from PIL import Image, ImageEnhance

import streamlit as st
import numpy as np
import cv2
import os


# define mask_detection of streamlit
def mask_detection():
    # load pre-trained face detector network 
    prototxtPath = "./Model/FaceNet/deploy.prototxt"
    weightsPath = "./Model/FaceNet/res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load face-mask detector model
    maskNet = load_model("./Model/MaskNet/mask_net.model")

    # load input image with heights & weights
    image = cv2.imread("./Intermediate_Pic.jpg")
    if image is None:
        return None

    # convert channel BGR -> RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    (h, w) = image.shape[:2]        # h => heights; w => weights

    # construct a blob of image
    blob = cv2.dnn.blobFromImage(
        image,
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0))

    # obtain face detections through blob & face detector network
    faceNet.setInput(blob)
    face_detections = faceNet.forward()

    # loop over face detections
    for i in range(0, face_detections.shape[2]):
        # extract confidence of face detection
        face_confidence = face_detections[0, 0, i, 2]
        
        if face_confidence > 0.50:
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
                color = (0, 250, 0)     # green in RGB
            else:
                color = (250, 0, 0)     # red in RGB

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

    return image



# set custom page state of title, layout, sidebar
st.set_page_config(
    page_title = "Simplified Face-Mask Detector",
    layout = "centered",
    initial_sidebar_state = "expanded"
)

# set markdown flavored content
st.markdown(
    '<h1 align = "center">Simplified Face-Mask Detection</h1>',
    unsafe_allow_html = True
)

# set input stream's option
activities = ["Image", "Webcam"]

# set input choice
choice = st.sidebar.selectbox("Choose among the input stream's type: ", activities)

if choice == "Image":
    st.markdown(
        '<h2 align = "center">Detections on Images</h2>',
        unsafe_allow_html = True
    )
    st.markdown(
        "### Upload your image here: "
    )

    # upload image file
    image_file = st.file_uploader("", type=['jpg'])

    # start detection if source img is available
    if image_file is not None:
        src_image = Image.open(image_file)
        src_image.save("./Intermediate_Pic.jpg")

        # display uploaded image
        st.image(
            src_image,
            caption = "Uploaded Source Image",
            use_column_width=True
        )
        
        # display notation
        st.markdown(
            '<h3 align="center">Image uploaded successfully!</h3>',
            unsafe_allow_html=True
        )

        if st.button("Start Prediction"):
            st.image(mask_detection(), use_column_width = True)
            os.remove("./Intermediate_Pic.jpg")
            st.markdown(
            '<h3 align="center">Image detected successfully!</h3>',
            unsafe_allow_html=True
            )

if choice == "Webcam":
    st.markdown(
        '<h2 align = "center">Detections on Webcam Video Streams</h2>',
        unsafe_allow_html = True
    )

    st.markdown(
        '<h2 align = "center">Hot work in progress...</h2>',
        unsafe_allow_html = True
    )