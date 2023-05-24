
import numpy as np
from PIL import Image
from tensorflow.keras.utils import to_categorical
from city_sustainability.preprocessing import image_resize

import numpy as np
from PIL import Image
from keras.utils import to_categorical

def image_and_label_arrays_batch(image_paths, label_paths, sampling_ratio=1, batch_size=None):
    # Default sampling_ratio is equal to 1.0 (selects all samples)
    if sampling_ratio < 1.0:
        end = round(sampling_ratio * len(image_paths))
        image_paths = image_paths[:end]
        label_paths = label_paths[:end]

    if batch_size is None:
        image_list_array = []
        label_list_array = []
    else:
        image_list_array = [[]]
        label_list_array = [[]]
        batch_count = 0

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

        if batch_size is None:
            image_list_array.append(numpy_array_image)
            label_list_array.append(encoded_label)
        else:
            image_list_array[batch_count].append(numpy_array_image)
            label_list_array[batch_count].append(encoded_label)

            if (idx + 1) % batch_size == 0:
                image_list_array.append([])
                label_list_array.append([])
                batch_count += 1

    if batch_size is None:
        image_array = np.array(image_list_array)  # Convert the list of arrays to a NumPy array
        label_array = np.array(label_list_array)  # Convert the list of arrays to a NumPy array
    else:
        image_array = [np.array(batch) for batch in image_list_array if batch]
        label_array = [np.array(batch) for batch in label_list_array if batch]

    return image_array, label_array
