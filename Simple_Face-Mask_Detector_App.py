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
import streamlit as st

from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils import img_to_array
from keras.models import load_model

import pandas as pd
import numpy as np
import cv2
import os
import urllib



def main():
    """
    Starting `streamlit` execution in a main() function.
    """

    # render the pre-given markdown file in page
    README_TEXT = st.markdown(load_file_content_as_string("App_Instruction.md"))

    # download external dependencies
    for fileName in EXTERNAL_DEPENDENCIES.keys():
        download_file(fileName)

    # render the selector sidebar once complete downloading
    st.sidebar.title("What to do")
    selectMode = st.sidebar.selectbox(
        "Choose the detection mode",
        ["Display Sample Usage", "Display Source Code", "Detect on Image", "Detect on Webcam Video Stream"])

    if selectMode == "Disply Sample Usage":
        st.sidebar.success("To continue on Face-Mask Detection, select `Detect on Image` or `Detect on Webcam Video Stream`")
    
    elif selectMode == "Display Source Code":
        st.code(load_file_content_as_string("Simple_Face-Mask_Detector_App.py"))
    
    elif selectMode == "Detect on Image":
        README_TEXT.empty()
        face_mask_detection_image()
    
    elif selectMode == "Detect on Webcam Video Stream":
        st.sidebar.warning("Streamlit doesn't currently have any browser-based camera support.")
        st.sidebar.info("Therefore further development of this feature has been postponed indefinitely.")



def download_file(path):
    # verify existence
    if not os.path.exists(path):
        return False

    # initialize visual components to animate
    weights_warning = None
    progress_bar = None

    # set animation
    try:
        weights_warning = st.warning("Downloading %s..." % path)
        progress_bar = st.progress(0)

        with open(path, "wb") as output:
            with urllib.request.urlopen(EXTERNAL_DEPENDENCIES[path]["url"]) as response:
                length = int(response.info(["Content-Length"]))
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0         # 2 ^ 20
                
                while True:
                    data = response.read(8192)      # Max Byte-Array Buffer Read at one time
                    if not data:
                        break                       # Load Compelete
                    counter += len(data)
                    output.write(data)

                    # operate animation by overwriting components
                    weights_warning("Downloading %s...(%6.2f/%6.2f MB)" % 
                        (path, counter / MEGABYTES, length / MEGABYTES))

    # clear all components after downloading
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        
        if progress_bar is not None:
            progress_bar.empty()

    return True



def face_mask_detection_image():
    st.markdown("")




@st.experimental_singleton(show_spinner=False)
def load_file_content_as_string(path):
    repo_url = "https://github.com/Ch-i-Yu/Simple-Face-Mask-Detector/blob/main" + "/" + path
    response = urllib.request.urlopen(repo_url)
    return response.read().decode("utf-8")



# External files to download
EXTERNAL_DEPENDENCIES = {
    "deploy.prototxt":{
        "url": "https://github.com/Ch-i-Yu/Simple-Face-Mask-Detector/blob/main/Model/FaceNet/deploy.prototxt"
    },

    "res10_300x300_ssd_iter_140000.caffemodel":{
        "url": "https://github.com/Ch-i-Yu/Simple-Face-Mask-Detector/blob/main/Model/FaceNet/res10_300x300_ssd_iter_140000.caffemodel"
    },

    "mask_net.model":{
        "url": "https://github.com/Ch-i-Yu/Simple-Face-Mask-Detector/blob/main/Model/MaskNet/mask_net.model"
    }
}