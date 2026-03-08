import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import class_weight
from preprocess import load_and_preprocess


def build_model(input_dim):

    model = keras.Sequential([
        keras.Input(shape=(input_dim,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    return model


def train_model():

    X_train, X_test, y_train, y_test = load_and_preprocess()

    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )

    class_weights_dict = {
        0: class_weights[0],
        1: class_weights[1]
    }

    model = build_model(X_train.shape[1])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_test, y_test),
        class_weight=class_weights_dict
    )

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "ann_model.h5")

    model.save(model_path)

    print("Model saved successfully!")


if __name__ == "__main__":
    train_model()