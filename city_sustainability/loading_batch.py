import numpy as np
from PIL import Image
from keras.utils import to_categorical

def image_and_label_arrays_batch(image_paths, label_paths, sampling_ratio=1, batch_size=32):
    # Default sampling_ratio is equal to 1.0 (selects all samples)
    if sampling_ratio < 1.0:
        end = round(sampling_ratio * len(image_paths))
        image_paths = image_paths[:end]
        label_paths = label_paths[:end]

    image_list_array = []
    label_list_array = []

    batch_images = []
    batch_labels = []

    for idx, (image_path, label_path) in enumerate(zip(image_paths, label_paths)):
        im = Image.open(image_path)
        lb = Image.open(label_path)

        # Resize each image and label using image_resize function
        resized_image = image_resize(256, 256, im)
        resized_label = image_resize(256, 256, lb)

        # Generate array for each image and label
        numpy_array_image = np.array(resized_image)
        numpy_array_label = np.array(resized_label)

        # Encode labels
        encoded_label = to_categorical(numpy_array_label, num_classes=9)

        batch_images.append(numpy_array_image)
        batch_labels.append(encoded_label)

        if len(batch_images) == batch_size:
            image_list_array.append(np.array(batch_images))
            label_list_array.append(np.array(batch_labels))
            batch_images = []
            batch_labels = []

    if batch_images:
        image_list_array.append(np.array(batch_images))
        label_list_array.append(np.array(batch_labels))

    image_array = np.concatenate(image_list_array, axis=0)  # Concatenate the list of arrays into a single array
    label_array = np.concatenate(label_list_array, axis=0)  # Concatenate the list of arrays into a single array

    return image_array, label_array
