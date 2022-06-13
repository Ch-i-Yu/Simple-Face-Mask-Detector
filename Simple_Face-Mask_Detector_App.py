# -*- coding: utf-8 -*-



"""
Face-Mask Detector with OpenCV and Keras

This goal of this project is to roughly identify
whether a person is wearing a mask or not.

This project aims to:
- Train a model on images of people wearing masks
- Deploy the trained model to faces-masks in images and video streams

This python program aims to:
- Deploy a model to detect faces-masks in images with Streamlit App

"""



# import required dependencies and libraries
from ast import Global
from importlib_metadata import FreezableDefaultDict
import streamlit as st

from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils import img_to_array
from keras.models import load_model

from PIL import Image, ImageEnhance

import pandas as pd
import numpy as np
import cv2
import os
import urllib
import time



def main():
    """
    Starting `streamlit` execution in a main() function.
    """
    # set page title, layout, sidebar
    st.set_page_config(page_title = "Simple Face-Mask Detector")

    DOWNLOAD_TEXT = st.markdown("### Loading...Please wait")

    # download external dependencies
    for fileName in EXTERNAL_DEPENDENCIES.keys():
        download_file(fileName)

    DOWNLOAD_TEXT.empty()

    # render the pre-given markdown file in page
    README_TEXT = st.markdown(load_file_content_as_string("App_Instruction.md"))

    # render the selector sidebar once complete downloading
    st.sidebar.title("What to do")
    selectMode = st.sidebar.selectbox(
        "Choose the detection mode",
        ["Display Sample Usage", "Display Source Code", "Detect on Image", "Detect on Webcam Video Stream"])

    if selectMode == "Disply Sample Usage":
        st.sidebar.success("To continue on Face-Mask Detection, select `Detect on Image` or `Detect on Webcam Video Stream`")
    
    elif selectMode == "Display Source Code":
        README_TEXT.empty()
        st.code(load_file_content_as_string("Simple_Face-Mask_Detector_App.py"))
    
    elif selectMode == "Detect on Image":
        README_TEXT.empty()
        execute()
    
    elif selectMode == "Detect on Webcam Video Stream":
        st.sidebar.warning("Streamlit doesn't currently have any browser-based camera support.")
        st.sidebar.info("Therefore further development of this feature has been postponed indefinitely.")



def download_file(fileName):
    # initialize visual components to animate
    weights_warning = None
    progress_bar = None

    # set animation
    try:
        dst_folder = os.path.exists(EXTERNAL_DEPENDENCIES[fileName]["directory"])
        if not dst_folder:
            os.makedirs(EXTERNAL_DEPENDENCIES[fileName]["directory"])

        dst_path = EXTERNAL_DEPENDENCIES[fileName]["directory"] + "/" + fileName

        if os.path.exists(dst_path):
            # skip downloading if dependencies already exists
            return

        weights_warning = st.warning("Downloading %s..." % fileName)
        progress_bar = st.progress(0)

        with open(dst_path, "wb") as output:
            with urllib.request.urlopen(EXTERNAL_DEPENDENCIES[fileName]["url"]) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0         # 2 ^ 20
                
                while True:
                    data = response.read(8192)      # Max Byte-Array Buffer Read at one time
                    if not data:
                        break                       # Load Compelete
                    counter += len(data)
                    output.write(data)

                    # operate animation by overwriting components
                    weights_warning.warning("Downloading %s...(%6.2f/%6.2f MB)" % 
                        (fileName, counter / MEGABYTES, length / MEGABYTES))
                    progress_bar.progress(min(counter / length, 1.0))

    # clear all components after downloading
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        
        if progress_bar is not None:
            progress_bar.empty()

    return



def execute():
    def face_mask_detect(origin_image):
        # load pre-trained face detector DNN net
        faceNet = cv2.dnn.readNet("Model/FaceNet/deploy.prototxt", "Model/FaceNet/res10_300x300_ssd_iter_140000.caffemodel")

        # load pre-trained mask detector h5 net
        maskNet = load_model("Model/MaskNet/mask_net.model")

        # load input image with dimensions
        (h, w) = origin_image.shape[:2]

        # construct a blob from the image
        blob = cv2.dnn.blobFromImage(origin_image, 1.0, (300, 300), (104.0, 177.0, 123.0))

        # obtain face detections through blob & face detector network
        faceNet.setInput(blob)
        face_detections = faceNet.forward()

        # loop over face detections
        for i in range(0, face_detections.shape[2]):
            # extract confidence of face detection
            face_confidence = face_detections[0, 0, i, 2]
            
            if face_confidence > 0.5:
                print("<Process Face-Mask Detections......>")

                # compute the (x, y) coordinates of the bounding box of detected face
                face_box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x_start, y_start, x_end, y_end) = face_box.astype("int")

                # adjust the bounding box to fall within dimensions of the frame(image)
                (x_start, y_start) = (max(0, x_start), max(0, y_start))
                (x_end, y_end) = (max(0, x_end), max(0, y_end))

                # extract face ROI
                face = origin_image[y_start:y_end, x_start:x_end]
                
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
                    img = origin_image,
                    text = label,
                    org = (x_start, y_start-10),
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale = 0.65,
                    color = color,
                    thickness = 2
                )   #label
                cv2.rectangle(
                    img = origin_image, 
                    pt1 = (x_start, y_start),
                    pt2 = (x_end, y_end),
                    color = color, 
                    thickness = 3
                )   # bounding box

        return origin_image

    st.markdown('<h2 align="center">Detection on Image</h2>', unsafe_allow_html=True)
    st.markdown("**Upload your image here:**")

    uploaded_image = st.file_uploader("", type = ['jpg'])

    if uploaded_image is not None:
        st.image(
            uploaded_image,
            caption = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()),
            use_column_width = True)

        origin_image = np.array(Image.open(uploaded_image))
        
        st.markdown('<h3 align="center">Image uploaded successfully!</h3>', unsafe_allow_html=True)

        if st.button("Process"):
            dst_image = face_mask_detect(origin_image)
            st.image(
                dst_image,
                use_column_width = True)
            
            st.markdown('<h3 align="center">Image uploaded successfully!</h3>', unsafe_allow_html=True)



@st.experimental_singleton(show_spinner=False)
def load_file_content_as_string(path):
    repo_url = "https://raw.githubusercontent.com/Ch-i-Yu/Simple-Face-Mask-Detector/main" + "/" + path
    response = urllib.request.urlopen(repo_url)
    return response.read().decode("utf-8")



# External files to download
EXTERNAL_DEPENDENCIES = {
    "deploy.prototxt":{
        "url": "https://raw.githubusercontent.com/Ch-i-Yu/Simple-Face-Mask-Detector/main/Model/FaceNet/deploy.prototxt",
        "directory": "Model/FaceNet"
    },

    "res10_300x300_ssd_iter_140000.caffemodel":{
        "url": "https://raw.githubusercontent.com/Ch-i-Yu/Simple-Face-Mask-Detector/main/Model/FaceNet/res10_300x300_ssd_iter_140000.caffemodel",
        "directory": "Model/FaceNet"
    },

    "mask_net.model":{
        "url": "https://raw.githubusercontent.com/Ch-i-Yu/Simple-Face-Mask-Detector/main/Model/MaskNet/mask_net.model",
        "directory": "Model/MaskNet"
    }
}



if __name__ == "__main__":
    main()