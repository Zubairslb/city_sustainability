import streamlit as st
import pandas as pd
import numpy as np
import PIL
import matplotlib.pyplot as plt

# Title
st.title("Watch our model do some magic! Upload an image and get the quality of life prediction \U+1F609")

# Add some text
st.write("Our quality of life preduction first divides the classes into 3 metrics:")
st.write("1. Environmental_metric")
st.write("    Sum of the percentages of Rangeland, Tree, and Water")
st.write("2. Infrastructure_metric")
st.write("    Sum of the percentages of Developed Space, Road, and Building")
st.write("3. Land_metric")
st.write("    Sum of the percentages of Bareland, Agriculture land, and Other")
st.write("************")
st.write("Later on these metrics are classified into High, Medium and Low quality of life \U0001F929\U0001F929\U0001F929")

data_file = st.file_uploader(label='Upload an Image')