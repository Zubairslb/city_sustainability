from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    Dropout,
    MaxPooling2D,
    UpSampling2D,
    Concatenate,
    Input,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau


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


def build_model(input_shape=(28, 28, 1), num_classes=12):
    inputs = Input(input_shape)
    conv1, conv2, encoded = encoder(inputs)
    decoded = decoder(conv1, conv2, encoded)
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(decoded)
    return Model(inputs=inputs, outputs=outputs)


def compile_model(model, optimizer='adam'):
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


def train_model(model, x, y, epochs=1, batch_size=32, validation_split=0.1):
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001, verbose=1)
    model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[lr_reducer])


def evaluate_model(model, x, y):
    loss, accuracy = model.evaluate(x, y)
    print("Evaluation results:")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")


def predict(model, x):
    return model.predict(x)