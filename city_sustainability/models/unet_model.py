import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    Dropout,
    MaxPooling2D,
    UpSampling2D,
    Concatenate,
    Input,)

def encoder(inputs):
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Dropout(0.1)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Dropout(0.1)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    return conv1, conv2, pool2


def decoder(conv1, conv2, encoded):
    up1 = UpSampling2D(size=(2, 2))(encoded)
    merge1 = Concatenate(axis=3)([conv2, up1])
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge1)
    conv3 = BatchNormalization()(conv3)
    conv3 = Dropout(0.1)(conv3)

    up2 = UpSampling2D(size=(2, 2))(conv3)
    merge2 = Concatenate(axis=3)([conv1, up2])
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge2)
    conv4 = BatchNormalization()(conv4)
    conv4 = Dropout(0.1)(conv4)

    return conv4


def decoder_full(conv1, conv2, conv3, conv4, conv5, encoded):
    
    up5 = UpSampling2D(size=(2, 2))(encoded)
    merge5 = Concatenate(axis=3)([conv5, up5])
    l5 = Conv2D(512, (3, 3), activation='relu', padding='same')(merge5)
    l5 = BatchNormalization()(l5)
    l5 = Dropout(0.1)(l5) 
    
    up4 = UpSampling2D(size=(2, 2))(l5)
    merge4 = Concatenate(axis=3)([conv4, up4])
    l4 = Conv2D(512, (3, 3), activation='relu', padding='same')(merge4)
    l4 = BatchNormalization()(l4)
    l4 = Dropout(0.1)(l4) 
    
    up3 = UpSampling2D(size=(2, 2))(l4)
    merge3 = Concatenate(axis=3)([conv3, up3])
    l3 = Conv2D(256, (3, 3), activation='relu', padding='same')(merge3)
    l3 = BatchNormalization()(l3)
    l3 = Dropout(0.1)(l3)   
    
    up2 = UpSampling2D(size=(2, 2))(l3)
    merge2 = Concatenate(axis=3)([conv2, up2])
    l2 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge2)
    l2 = BatchNormalization()(l2)
    l2 = Dropout(0.1)(l2)

    up1 = UpSampling2D(size=(2, 2))(l2)
    merge1 = Concatenate(axis=3)([conv1, up1])
    l1 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge1)
    l1 = BatchNormalization()(l1)
    l1 = Dropout(0.1)(l1)

    return l1


def build_model(input_shape=(28, 28, 1), num_classes=12):
    inputs = Input(input_shape)
    conv1, conv2, encoded = encoder(inputs)
    decoded = decoder(conv1, conv2, encoded)
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(decoded)
    return Model(inputs=inputs, outputs=outputs)


def build_vgg16_model(input_shape=(28, 28, 1), num_classes=12 ):
    model_vgg = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    for layer in model_vgg.layers:
        layer.trainable = False
    inputs = model_vgg.layers[0].output
    conv1 = model_vgg.get_layer('block1_conv2').output
    conv2 = model_vgg.get_layer('block2_conv2').output
    encoded = model_vgg.get_layer('block3_conv3').output
    decoded = decoder(conv1, conv2, encoded)
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(decoded)
    return Model(inputs=inputs, outputs=outputs)


def build_vgg16_model_full(input_shape=(28, 28, 1), num_classes=12):
    model_vgg = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    for layer in model_vgg.layers:
        layer.trainable = False
    inputs = model_vgg.layers[0].output
    conv1 = model_vgg.get_layer('block1_conv2').output
    conv2 = model_vgg.get_layer('block2_conv2').output
    conv3 = model_vgg.get_layer('block3_conv3').output
    conv4 = model_vgg.get_layer('block4_conv3').output
    conv5 = model_vgg.get_layer('block5_conv3').output
    encoded = model_vgg.get_layer('block5_pool').output       
    decoded = decoder_full(conv1, conv2, conv3, conv4, conv5, encoded)   
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(decoded)    
    return Model(inputs=inputs, outputs=outputs)


def compile_model(model, optimizer='adam'):
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

def compute_iou(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3]) - intersection
    iou = tf.reduce_mean((intersection + 1e-7) / (union + 1e-7))
    return iou




def train_model(model, x, y, epochs=1, batch_size=32, validation_split=0.1, class_balance=False):
    if class_balance:
        # Reshape y to have 1 dimension
        y_flat = np.argmax(y, axis=-1)
        class_labels = np.unique(y_flat)
        
        # Count the pixels in each class
        class_counts = np.bincount(y_flat.flatten())
        
        # Calculate class weights
        class_weights = np.max(class_counts) / class_counts
        
        # Generate pixel-wise sample weights based on class weights
        sample_weights = np.take(class_weights, y_flat)
        
    else:
        sample_weights = None
    
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy', compute_iou], weighted_metrics=[])
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001, verbose=1)
    early_stopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    
    history = model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split,
                        callbacks=[lr_reducer, early_stopper], sample_weight=sample_weights)
    
    return history


<<<<<<< HEAD

=======
>>>>>>> 9f95d5eb3601158c49b31f39d5e4eaddb9c48bcb
def evaluate_model(model, x, y):
    loss, accuracy, iou = model.evaluate(x, y)
    
    # Calculate IoU
    y_pred = model.predict(x)
    calculated_iou = compute_iou(y, y_pred)
    
    print("Evaluation results:")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Model IoU: {iou:.4f}")

    if calculated_iou is not None:
        print(f"Calculated IoU: {calculated_iou:.4f}")
    else:
        print("Calculated IoU: Not Available")
    
    return loss


def predict(model, x):
    return model.predict(x)
