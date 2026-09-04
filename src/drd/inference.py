from pathlib import Path
import numpy as np
import tensorflow as tf
from . import CLASS_NAMES
from .preprocessing import load_image


def load_model(path: str | Path):
    return tf.keras.models.load_model(path)

def predict_image(model, image_path: str | Path, image_size=(299, 299)):
    batch = np.expand_dims(load_image(image_path, image_size), axis=0)
    probabilities = model.predict(batch, verbose=0)[0]
    index = int(np.argmax(probabilities))
    return {
        "predicted_class": CLASS_NAMES[index],
        "confidence": float(probabilities[index]),
        "probabilities": {name: float(probabilities[i]) for i, name in enumerate(CLASS_NAMES)},
    }
