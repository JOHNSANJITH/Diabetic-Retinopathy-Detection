from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from .preprocessing import load_image


def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer.name
        except Exception:
            pass
    raise ValueError("Could not find a convolutional feature layer")


def gradcam(model, image_path, output_path, image_size=(299, 299), layer_name=None):
    layer_name = layer_name or find_last_conv_layer(model)
    feature_model = tf.keras.Model(model.inputs, [model.get_layer(layer_name).output, model.output])
    image = load_image(image_path, image_size)
    batch = np.expand_dims(image, 0)
    with tf.GradientTape() as tape:
        features, predictions = feature_model(batch)
        class_index = tf.argmax(predictions[0])
        score = predictions[:, class_index]
    gradients = tape.gradient(score, features)
    weights = tf.reduce_mean(gradients, axis=(1, 2))
    cam = tf.reduce_sum(features * weights[:, None, None, :], axis=-1)[0]
    cam = tf.maximum(cam, 0)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    heatmap = np.uint8(255 * cam.numpy())

    original = Image.open(image_path).convert("RGB")
    heatmap_img = Image.fromarray(heatmap).resize(original.size)
    heatmap_arr = np.asarray(heatmap_img) / 255.0
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 2, 1); ax.imshow(original); ax.axis("off"); ax.set_title("Fundus image")
    ax = fig.add_subplot(1, 2, 2); ax.imshow(original); ax.imshow(heatmap_arr, cmap="jet", alpha=0.45); ax.axis("off"); ax.set_title("Grad-CAM")
    out = Path(output_path); out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
    return int(class_index), str(out), layer_name
