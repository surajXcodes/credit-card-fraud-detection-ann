import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("models/ann_model.h5")

def predict_transaction(data):

    data = np.array(data).reshape(1, -1)

    prediction = model.predict(data)

    if prediction > 0.5:
        print("Fraud Transaction 🚨")
    else:
        print("Genuine Transaction ✅")