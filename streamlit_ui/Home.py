import streamlit as st
import pandas as pd
import numpy as np
import PIL
import matplotlib.pyplot as plt


st.title("Welcome to our City Sustainability Platform​ :smile:")

st.write("## What is City Sustainability? :city_sunrise:")
st.write("#### City sustainability refers to the concept of creating and maintaining cities that are environmentally, socially, and economically sustainable. It involves developing urban areas that minimize their impact on the environment, promote social equity, and support economic prosperity.")

# Provide the path or URL of the image
image = "Images/City.jpg"
# or image_url = "https://example.com/image.jpg"

# Display the image
st.image(image)

st.write("************")

st.write("## Our Mission :star2:")
st.write("#### Identify land occupation to compute quality of life metrics for a city")

st.write("************")

st.write("## Model Background :eyes:")
st.write("#### Our model follows a UNet onvolutional neural network (CNN) architecture. UNet was originally invented and first used for biomedical image segmentation. Its architecture can be broadly thought of as an encoder network followed by a decoder network.")
st.write("- The encoder is the first half in the architecture diagram. It usually is a pre-trained classification network like VGG/ResNet where you apply convolution blocks followed by a maxpool downsampling to encode the input image into feature representations at multiple different levels.")
st.write(" - The decoder is the second half of the architecture. The goal is to semantically project the discriminative features (lower resolution) learnt by the encoder onto the pixel space (higher resolution) to get a dense classification. The decoder consists of upsampling and concatenation followed by regular convolution operations.")

# Provide the path or URL of the image
image_path = "Images/UNet Model.PNG"
# or image_url = "https://example.com/image.jpg"

# Display the image
st.image(image_path, use_column_width=True)

st.write("************")

st.write("## Dataset :books:")
st.write("#### The Dataset used to train our model consists of certain classes and annotations as shown below:")

# Provide the path or URL of the image
image_path_2 = "Images/Dataset Classes.png"
# or image_url = "https://example.com/image.jpg"

# Display the image
st.image(image_path_2, use_column_width=True)

st.write("#### Our model will classify each image into the classes above and will return a labeled image. From there, we will walk you through our Quality of Life classification.")
st.write("#### Explore the tabs on the left :point_left:")


st.write("************")

st.write("## About Us :raised_hand:")
st.write("#### We are a group of SLB engineers coming from various backgrounds and united in our interest in Data Science :computer:")

# Provide the path or URL of the image
image_path_3 = "Images/Logos.PNG"
# or image_url = "https://example.com/image.jpg"

# Display the image
st.image(image_path_3)
