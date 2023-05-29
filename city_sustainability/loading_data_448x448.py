import os

def loading_paths_448(data_path):
    # data_path is the path where all the city folders are located
    city_folders = os.listdir(data_path)
    
    # Initialize empty lists for images and labels
    image_paths = []
    label_paths = []
    
    # Iterate over city folders
    for city_folder in city_folders:
        city_path = os.path.join(data_path, city_folder)
        # Check if the item is a directory
        if os.path.isdir(city_path):
            image_folder = os.path.join(city_path, "images")
            label_folder = os.path.join(city_path, "labels")
            # Check if image and label folders exist
            if os.path.exists(image_folder) and os.path.exists(label_folder):
                # List all image files
                image_files = os.listdir(image_folder)
                # Iterate over image files
                for image_file in image_files:
                    image_path = os.path.join(image_folder, image_file)    
                    label_path = os.path.join(label_folder, image_file)
                    if os.path.exists(label_path):
                        # Append image and label file paths to lists
                        image_paths.append(image_path)
                        label_paths.append(label_path)
                        
    return image_paths, label_paths


import numpy as np
from PIL import Image
from tensorflow.keras.utils import to_categorical
from city_sustainability.preprocessing import image_resize

def image_and_label_arrays_448(image_paths, label_paths, sampling_ratio = 1):
    # Default sampling_ratio is equal to 1.0 (selects all samples)
    image_list_array = []
    label_list_array = []
    end = round(sampling_ratio*len(image_paths))

    for image_path in image_paths[0:end]:
        im = Image.open(image_path)
        # Resize each image using image_resize function
        resized_image = image_resize(448,448,im)
        # Generate array for each image
        numpy_array_image = np.array(resized_image)
        # Add resized array to list
        image_list_array.append(numpy_array_image)

    for label_path in label_paths[0:end]:
        lb = Image.open(label_path)
        # Resize each label using image_resize function
        resized_label = image_resize(448,448,lb)
        # Generate array for each image
        numpy_array_label = np.array(resized_label)
        # Encode labels
        encoded_label = to_categorical(numpy_array_label, num_classes=9)
        # Add resized and encoded array to list
        label_list_array.append(encoded_label)
    
    image_array = np.array(image_list_array)  # Convert the list of arrays to a NumPy array
    label_array = np.array(label_list_array)  # Convert the list of arrays to a NumPy array
    
    return image_array, label_array
