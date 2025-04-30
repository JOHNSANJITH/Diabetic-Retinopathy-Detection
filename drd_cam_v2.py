import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing.image import img_to_array

class_names = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']

model = tf.keras.models.load_model('diabetic_retinopathy_xception_model.h5')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def preprocess_image(image):
    img = cv2.resize(image, (299, 299))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def generate_gradcam(model, image_array, layer_name="block14_sepconv2_act"):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        predicted_class = np.argmax(predictions[0])
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1

    return heatmap, predicted_class, predictions

def draw_labeled_boxes(image, heatmap, predicted_label, threshold=0.5):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    _, binary_map = cv2.threshold((heatmap * 255).astype(np.uint8), int(255 * threshold), 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return image

frame_skip = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_skip += 1
    if frame_skip % 2 == 0:
        continue

    preprocessed_frame = preprocess_image(frame)
    heatmap, predicted_class, predictions = generate_gradcam(model, preprocessed_frame)

    label = class_names[predicted_class]
    frame_with_boxes = draw_labeled_boxes(frame, heatmap, label)

    confidence = np.max(predictions) * 100
    cv2.putText(frame_with_boxes, f"Prediction: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame_with_boxes, f"Confidence: {confidence:.2f}%", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Diabetic Retinopathy Detection', frame_with_boxes)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    