import tensorflow as tf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_WEIGHTS = BASE_DIR / "model.h5"


def build_model():
    model = tf.keras.Sequential(name="sequential_1")
    model.add(tf.keras.layers.Conv2D(16, (7, 7), padding="same", input_shape=(48, 48, 1), activation="linear", name="image_array"))
    model.add(tf.keras.layers.BatchNormalization(name="batch_normalization_1"))
    model.add(tf.keras.layers.Conv2D(16, (7, 7), padding="same", activation="linear", name="conv2d_1"))
    model.add(tf.keras.layers.BatchNormalization(name="batch_normalization_2"))
    model.add(tf.keras.layers.Activation("relu", name="activation_1"))
    model.add(tf.keras.layers.AveragePooling2D((2, 2), padding="same", name="average_pooling2d_1"))
    model.add(tf.keras.layers.Dropout(0.5, name="dropout_1"))
    model.add(tf.keras.layers.Conv2D(32, (5, 5), padding="same", activation="linear", name="conv2d_2"))
    model.add(tf.keras.layers.BatchNormalization(name="batch_normalization_3"))
    model.add(tf.keras.layers.Conv2D(32, (5, 5), padding="same", activation="linear", name="conv2d_3"))
    model.add(tf.keras.layers.BatchNormalization(name="batch_normalization_4"))
    model.add(tf.keras.layers.Activation("relu", name="activation_2"))
    model.add(tf.keras.layers.AveragePooling2D((2, 2), padding="same", name="average_pooling2d_2"))
    model.add(tf.keras.layers.Dropout(0.5, name="dropout_2"))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="linear", name="conv2d_4"))
    model.add(tf.keras.layers.BatchNormalization(name="batch_normalization_5"))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="linear", name="conv2d_5"))
    model.add(tf.keras.layers.BatchNormalization(name="batch_normalization_6"))
    model.add(tf.keras.layers.Activation("relu", name="activation_3"))
    model.add(tf.keras.layers.AveragePooling2D((2, 2), padding="same", name="average_pooling2d_3"))
    model.add(tf.keras.layers.Dropout(0.5, name="dropout_3"))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="linear", name="conv2d_6"))
    model.add(tf.keras.layers.BatchNormalization(name="batch_normalization_7"))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="linear", name="conv2d_7"))
    model.add(tf.keras.layers.BatchNormalization(name="batch_normalization_8"))
    model.add(tf.keras.layers.Activation("relu", name="activation_4"))
    model.add(tf.keras.layers.AveragePooling2D((2, 2), padding="same", name="average_pooling2d_4"))
    model.add(tf.keras.layers.Dropout(0.5, name="dropout_4"))
    model.add(tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation="linear", name="conv2d_8"))
    model.add(tf.keras.layers.BatchNormalization(name="batch_normalization_9"))
    model.add(tf.keras.layers.Conv2D(7, (3, 3), padding="same", activation="linear", name="conv2d_9"))
    model.add(tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling2d_1"))
    model.add(tf.keras.layers.Activation("softmax", name="predictions"))
    return model


def load_model():
    model = build_model()
    model.load_weights(MODEL_WEIGHTS)
    return model
