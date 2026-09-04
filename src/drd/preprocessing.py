from pathlib import Path
import numpy as np
from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

def load_image(path: str | Path, image_size=(299, 299)) -> np.ndarray:
    """Load RGB fundus image and apply Xception preprocessing."""
    image = Image.open(path).convert("RGB").resize(tuple(image_size))
    array = np.asarray(image, dtype=np.float32)
    # Xception/ImageNet preprocessing: [0, 255] RGB -> [-1, 1].
    return (array / 127.5) - 1.0

def load_batch(paths, image_size=(299, 299)) -> np.ndarray:
    return np.stack([load_image(path, image_size) for path in paths])

def build_augmentation():
    import tensorflow as tf
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomTranslation(0.05, 0.05),
    ], name="retinal_augmentation")
