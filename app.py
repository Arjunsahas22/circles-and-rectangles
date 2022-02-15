# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 12:45:46 2022

@author: Arjun sahas
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
st.write("""
          # Identifying circle or rectangle
          """
          )
upload_file = st.sidebar.file_uploader("Upload Images", type="jpg")
Generate_pred=st.sidebar.button("Predict")
model=tf.keras.models.load_model('figures.h5')
def prediction(image_data, model):
    size = (128,128)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    reshape=img[np.newaxis,...]
    pred = model.predict(reshape)
    return pred
if Generate_pred:
    image=Image.open(upload_file)
    with st.expander('Figure Image', expanded = True):
        st.image(image, use_column_width=True)
    pred=prediction(image, model)
    labels = ['Circle', 'Rectangle']
    st.title("Prediction of image is {}".format(labels[np.argmax(pred)]))