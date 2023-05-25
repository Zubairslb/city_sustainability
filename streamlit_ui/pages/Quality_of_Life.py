import streamlit as st
import pandas as pd
import numpy as np
import PIL
import matplotlib.pyplot as plt
from city_sustainability.quality import life_quality
from PIL import Image
from tensorflow.keras.utils import to_categorical
from city_sustainability.preprocessing import image_resize

# Title
st.title("Watch our model do some magic!! Upload an image and get the quality of life prediction :)")

# Add some text
st.write("Our quality of life preduction first divides the classes into 3 metrics:")
st.write("1. Environmental_metric")
st.write("    Sum of the percentages of Rangeland, Tree, and Water")
st.write("2. Infrastructure_metric")
st.write("    Sum of the percentages of Developed Space, Road, and Building")
st.write("3. Land_metric")
st.write("    Sum of the percentages of Bareland, Agriculture land, and Other")
st.write("************")
st.write("Later on these metrics are classified into High, Medium and Low quality of life")
st.write("************")

# Upload image
data_file = st.file_uploader(label='Upload an Image')

# Generate image
lb_1 = Image.open(data_file)

# Resize each label using image_resize function
resized_label = image_resize(256,256,lb_1)

# Generate array for each image
numpy_array_label = np.array(resized_label)

# Encode labels
encoded_label = to_categorical(numpy_array_label, num_classes=9)

# Run quality of life prediction
class_percentages, sorted_metrics, classification = life_quality(encoded_label)

# Display the image in Streamlit
st.image(lb_1)



#### Display class_percentage as pie-chart



# Extract the labels and values from the dictionary
labels = list(class_percentages.keys())
values = list(class_percentages.values())

# Create a pie chart
fig, ax = plt.subplots()
ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)

# Set aspect ratio to be equal so that pie is drawn as a circle
ax.axis('equal')

# Add a title to the pie chart
ax.set_title("Class distribution in the image")

# Move the legend to the side
ax.legend(labels, loc="center left", bbox_to_anchor=(1, 0.5))

# Display the pie chart in Streamlit
st.pyplot(fig)



#### Display sorted_metrics as a bie-chart



# Extract the labels and values from the sorted_metrics
labels_1 = [metric[0] for metric in sorted_metrics]
values_1 = [float(metric[1]) for metric in sorted_metrics]

# Define custom colors for the bars
colors = ['green', 'blue', 'brown']

# Create a bar chart
data = {'Metric': labels_1, 'Value': values_1}
chart = st.bar_chart(data=data, x='Metric', y='Value', color=colors)

# Rotate the x-axis labels
chart.set_xticklabels(labels_1, rotation=0)



#### Display final result
st.write("The model predicts:", classification)