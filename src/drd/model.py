import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from . import CLASS_NAMES


def build_model(image_size=(299, 299), dropout=0.35, learning_rate=1e-4, fine_tune=False, fine_tune_layers=30):
    base = Xception(weights="imagenet", include_top=False, input_shape=(*image_size, 3))
    base.trainable = False

    if fine_tune:
        base.trainable = True
        for layer in base.layers[:-fine_tune_layers]:
            layer.trainable = False

    inputs = tf.keras.Input(shape=(*image_size, 3), name="fundus_image")
    x = base(inputs, training=False)
    x = GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = Dropout(dropout, name="dropout")(x)
    outputs = Dense(len(CLASS_NAMES), activation="softmax", name="severity")(x)
    model = Model(inputs, outputs, name="drd_xception")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
