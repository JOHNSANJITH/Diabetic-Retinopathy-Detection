from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from . import CLASS_NAMES
from .preprocessing import build_augmentation


def _dataset(directory, image_size, batch_size, shuffle, seed, augmentation=None):
    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        image_size=tuple(image_size),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )
    if augmentation is not None:
        ds = ds.map(lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x, y: (tf.keras.applications.xception.preprocess_input(tf.cast(x, tf.float32)), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def make_datasets(data_dir, image_size, batch_size, seed):
    root = Path(data_dir)
    train = _dataset(root / "train", image_size, batch_size, True, seed, build_augmentation())
    validation = _dataset(root / "validation", image_size, batch_size, False, seed)
    test = _dataset(root / "test", image_size, batch_size, False, seed)
    return train, validation, test


def class_weights(train_dir):
    labels = []
    for index, name in enumerate(CLASS_NAMES):
        labels.extend([index] * len(list((Path(train_dir) / name).glob("*"))))
    if not labels:
        raise ValueError(f"No training images found under {train_dir}")
    classes = np.arange(len(CLASS_NAMES))
    weights = compute_class_weight("balanced", classes=classes, y=np.asarray(labels))
    return {int(k): float(v) for k, v in zip(classes, weights)}
