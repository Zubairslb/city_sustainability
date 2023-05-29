import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# train_image_directory = '/home/mgudipati/code/Zubairslb/city_sustainability/raw_data/resize_train_all/images',
# train_label_directory = '/home/mgudipati/code/Zubairslb/city_sustainability/raw_data/resize_train_all/labels',
# val_image_directory = '/home/mgudipati/code/Zubairslb/city_sustainability/raw_data/resize_val_all/images/'.
# val_label_directory = '/home/mgudipati/code/Zubairslb/city_sustainability/raw_data/resize_val_all/labels/',

def batching_from_dir(train_image_directory,
                      train_label_directory,
                      val_image_directory,
                      val_label_directory,
                      seed = 123,
                      batch_size=16,
                      num_classes= 9):

    image_datagen = ImageDataGenerator()
    mask_datagen = ImageDataGenerator()
    val_img_datagen = ImageDataGenerator()
    val_lab_datagen = ImageDataGenerator()

    seed = seed
    batch_size=batch_size


    image_generator = image_datagen.flow_from_directory(
        train_image_directory,
        class_mode=None,
        batch_size=batch_size,
        seed=seed,
        color_mode='rgb',
        target_size=(256, 256),
        subset = None,
        shuffle=False)

    mask_generator = mask_datagen.flow_from_directory(
        train_label_directory,
        class_mode=None,
        batch_size=batch_size,
        seed=seed,
        color_mode='grayscale',
        target_size=(256, 256),
        subset =  None,
        shuffle=False
    )

    val_image_generator = val_img_datagen.flow_from_directory(
        val_image_directory,
        class_mode=None,
        batch_size=batch_size,
        seed=seed,
        color_mode='rgb',
        target_size=(256, 256),
        subset = None,
        shuffle=False)

    val_mask_generator = val_lab_datagen.flow_from_directory(
        val_label_directory,
        class_mode=None,
        batch_size=batch_size,
        seed=seed,
        color_mode='grayscale',
        target_size=(256, 256),
        subset = None,
        shuffle=False
    )

    train_generator = zip(image_generator, mask_generator)
    val_generator = zip(val_image_generator, val_mask_generator)
    num_classes = num_classes

    def preprocess_labels(labels):
        categorical_labels = to_categorical(labels, num_classes=num_classes)
        return categorical_labels


    train_generator = ((images, preprocess_labels(labels)) for images, labels in train_generator)
    val_generator = ((images, preprocess_labels(labels)) for images, labels in val_generator)

    return train_generator, val_generator
