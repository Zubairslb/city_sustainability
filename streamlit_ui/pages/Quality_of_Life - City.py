import streamlit as st
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium
import math
import requests
import PIL
from city_sustainability.preprocessing import image_resize
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from city_sustainability.models.unet_model import compute_iou
import io
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from city_sustainability.quality import life_quality
import os

def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def create_map(latitude, longitude):
    # Create a Folium map centered at the specified latitude and longitude
    map_center = [latitude, longitude]
    m = folium.Map(
        location=map_center,
        zoom_start=17,
        tiles="https://api.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}@2x.jpg?access_token=pk.eyJ1IjoibWFsbWFocm9vcyIsImEiOiJjbGlhNnp1bnkwMHJpM3NzNmI0aW5zMGo1In0.I9GCCN-ZzW1RCfqw190w2g",
        attr="Mapbox attribution"
        )
    # Add a marker at the specified latitude and longitude
    folium.Marker(location=map_center).add_to(m)
    return m

# Streamlit app
st.title("Select any city around the world and get the Quality of Life classification :exclamation:")

st.write("## Enter any city below and use the interactive map to keep the desired area in the center")

# Create a geocoder
geolocator = Nominatim(user_agent="myGeocoder")

# Get the user input
city = st.text_input("")

if len(city) > 0:
    location = geolocator.geocode(city)
    latitude = location.latitude
    longitude = location.longitude
    map = create_map(latitude, longitude)
    st_data = st_folium(map)

    st.write("## Once satisfied, scroll below and witness the magic :boom:")

    st.write("************")

    st.write("## Below is a snapshot of the desired area and")

    # Get Satellite Image
    def get_satellite_image(latitude, longitude, map_zoom=17):
        x,y = deg2num(latitude, longitude, map_zoom)
        url = f'https://api.mapbox.com/v4/mapbox.satellite/{map_zoom}/{x}/{y}@2x.jpg?access_token=pk.eyJ1IjoibWFsbWFocm9vcyIsImEiOiJjbGlhNnp1bnkwMHJpM3NzNmI0aW5zMGo1In0.I9GCCN-ZzW1RCfqw190w2g'
        response = requests.get(url)
        return response.content

    latitude_1 = st_data["center"]["lat"]
    longitude_1 = st_data["center"]["lng"]
    zoom_1 = st_data["zoom"]

    data = get_satellite_image(latitude_1, longitude_1,zoom_1)

    # Generate image
    im_1 = Image.open(io.BytesIO(data))

    # Resize each label using image_resize function
    resized_im = image_resize(256,256,im_1)

    # Generate array for each image
    numpy_array_image = np.array(resized_im)

    # Scale the image
    numpy_array_image = numpy_array_image / 255

    # Expand image
    expanded_image = np.expand_dims(numpy_array_image, axis=0)

    # Load model
    path = os.path.dirname(__file__)
    model_path = path + "/../model/model_vgg16_s.h5"

    model = load_model(model_path,custom_objects={'compute_iou': compute_iou})

    # Run prediction on the image and generate label
    y_pred = model.predict(expanded_image)


    # Reshape label to appropriate size (remove the number of images)
    reshaped_pred = np.squeeze(y_pred)

    # remove the number of classes
    y_pred_np = np.argmax(reshaped_pred, axis=-1)

    # Display the image and labeled image (from model) in Streamlit


    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Display the first image
    axs[0].imshow(numpy_array_image)
    axs[0].set_title('Original Image')
    axs[0].axis('off')


    # Display the second image
    class_colors = {
            "Other": '#FFFF00',
            "Bareland": '#800000',
            "Rangeland": '#00FF24',
            "Developed space": '#949494',
            "Road": '#FFFFFF',
            "Tree": '#226126',
            "Water": '#0045FF',
            "Agriculture land": '#4BB549',
            "Building": '#DE1F07'
            }

    # Create a color map using the class_colors dictionary
    cmap = plt.cm.colors.ListedColormap(list(class_colors.values()))

    # Plot the image with the color map
    im = axs[1].imshow(y_pred_np, cmap=cmap, vmin=0, vmax=8)
    axs[1].set_title('Predicted Label')
    axs[1].axis('off')

    # Create a custom legend using the class_colors dictionary
    legend_elements = [mpatches.Patch(facecolor=color, edgecolor='black', label=class_label)
                        for class_label, color in class_colors.items()]

    # Add the legend outside the plot
    axs[1].legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.2)

    # Display the figure in Streamlit
    st.pyplot(fig)


    # Run quality of life prediction
    class_percentages, sorted_metrics, classification = life_quality(reshaped_pred)



    #### Display class_percentage as bar chart


    # Define the class colors dictionary
    class_colors = {
            "Other": '#FFFF00',   # Class 0 - Yellow
            "Bareland": '#800000',   # Class 1 - Maroon
            "Rangeland": '#00FF24',   # Class 2 - Lime Green
            "Developed space": '#949494',   # Class 3 - Grey
            "Road": '#FFFFFF',   # Class 4 - White
            "Tree": '#226126',   # Class 5 - Forest Green
            "Water": '#0045FF',   # Class 6 - Blue
            "Agriculture land": '#4BB549',   # Class 7 - Dark Green
            "Building": '#DE1F07'   # Class 8 - Red
            }

    # Extract the labels and values from the dictionary
    labels = list(class_percentages.keys())
    values = list(class_percentages.values())

    # Sort the labels and values in descending order based on values
    sorted_data = sorted(zip(values, labels), reverse=True)
    sorted_values, sorted_labels = zip(*sorted_data)

    # Create a bar chart with custom colors and black borders
    fig, ax = plt.subplots()
    bars = ax.bar(sorted_labels, sorted_values, color=[class_colors[label] for label in sorted_labels], edgecolor='black')

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

    # Create custom legend handles for each class
    legend_handles = [plt.bar(0, 0, color=class_colors[label], edgecolor='black')[0] for label in sorted_labels]

    # Display the legend on the right
    ax.legend(legend_handles, sorted_labels, loc='center left', bbox_to_anchor=(1, 0.5))

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

    # Define the metric colors dictionary
    metric_colors = {
        'Land': '#964B00',
        'Environmental': '#00FF00',
        'Infrastructure': '#0045FF'
    }

    # Extract the labels and values from the sorted_metrics
    labels_1 = [metric[0] for metric in sorted_metrics]
    values_1 = [float(metric[1]) for metric in sorted_metrics]

    # Get the colors from the metric_colors dictionary
    colors = [metric_colors[label] for label in labels_1]

    # Create a pie chart
    fig, ax = plt.subplots()
    ax.pie(values_1, labels=labels_1, colors=colors, autopct='%1.1f%%', startangle=90)

    # Set aspect ratio to be equal so that the pie is drawn as a circle
    ax.axis('equal')

    # Add a title to the pie chart
    ax.set_title('Metric distribution in the image')

    # Add a legend to the right
    ax.legend(labels_1, loc='best')

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

    st.write("************")
