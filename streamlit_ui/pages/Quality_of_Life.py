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
st.title("Watch our model do some magic!! Upload an image and get the quality of life classification :sunglasses:")

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

# Display the image in Streamlit
#fig = plt.imshow(lb_1)
#st.pyplot(fig)

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.imshow(lb_1)
st.pyplot(fig)

st.write("************")

# Run quality of life prediction
class_percentages, sorted_metrics, classification = life_quality(encoded_label)



#### Display class_percentage as bar chart


# Extract the labels and values from the dictionary
labels = list(class_percentages.keys())
values = list(class_percentages.values())

# Sort the labels and values in descending order based on values
sorted_data = sorted(zip(values, labels), reverse=True)
sorted_values, sorted_labels = zip(*sorted_data)

# Define a fixed color palette
colors = ['green', 'blue', 'brown', 'orange', 'purple', 'pink', 'gray', 'cyan', 'yellow']

# Create a bar chart with custom colors
fig, ax = plt.subplots()
bars = ax.bar(sorted_labels, sorted_values, color=colors)

# Add labels and values to the bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}%", ha='center', va='bottom')

# Set the title of the chart
ax.set_title("Class distribution in the image")

# Set the labels for x-axis and y-axis
ax.set_xlabel("Class")
ax.set_ylabel("Percentage")

# Rotate x-axis labels by 45 degrees
plt.xticks(rotation=45)

# Set the y-axis limits from 0 to 100
ax.set_ylim(0, 100)

# Remove the legend
ax.legend().remove()

# Display the bar chart in Streamlit
st.pyplot(fig)

st.write("************")

#### Display sorted_metrics as a bar-chart

# Add some text
st.write("## Our quality of life prediction divides the classes into 3 metrics:")
st.write("### 1. Environmental Metric :evergreen_tree:")
st.write("#####    Sum of the percentages of Rangeland, Tree, and Water")
st.write("### 2. Infrastructure Metric :city_sunrise:")
st.write("#####    Sum of the percentages of Developed Space, Road, and Building")
st.write("### 3. Land Metric :tent:")
st.write("#####    Sum of the percentages of Bareland, Agriculture land, and Other")
st.write("************")


# Extract the labels and values from the sorted_metrics
labels_1 = [metric[0] for metric in sorted_metrics]
values_1 = [float(metric[1]) for metric in sorted_metrics]

# Define custom colors for the pie chart
colors = ['brown', 'green', 'blue']

# Create a pie chart
fig, ax = plt.subplots()
ax.pie(values_1, labels=labels_1, colors=colors, autopct='%1.1f%%', startangle=90)

# Set aspect ratio to be equal so that the pie is drawn as a circle
ax.axis('equal')

# Add a title to the pie chart
ax.set_title('Metric distribution in the image')

# Add a legend to the right
ax.legend(labels_1, loc='center left', bbox_to_anchor=(1, 0.5))

# Display the pie chart in Streamlit
st.pyplot(fig)



st.write("## These metrics are used to classify the image into High, Medium and Low quality of life")
st.write("************")

#### Display final result
if classification == "Low quality of life":
    st.write("# Classification:", classification, ":disappointed:")
elif classification == "Medium quality of life":
    st.write("# Classification:", classification, ":expressionless:")
elif classification == "High quality of life":
    st.write("# Classification:", classification, ":satisfied:")
else:
    st.write("# Classification:", classification)