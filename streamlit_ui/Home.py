import streamlit as st
import pandas as pd
import numpy as np
import PIL
import matplotlib.pyplot as plt


st.title("Welcome to our city sustainability dashboard!!")
data_file = st.file_uploader(label='Upload an Image')
# st.balloons()
image = PIL.Image.open(data_file)
st.image(image, caption='Sunrise by the mountains')

