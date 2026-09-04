import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing.image import img_to_array

class_names = ['no_DR', 'mild_DR', 'moderate_DR', 'severe_DR', 'proliferative_DR']

model = tf.keras.models.load_model('diabetic_retinopathy_xception_model.h5')

cap = cv2.VideoCapture(0)

def preprocess_image(image):
    img = cv2.resize(image, (299, 299))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def detect_abnormalities(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
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
    predictions = model.predict(preprocessed_frame)
    predicted_class = np.argmax(predictions, axis=1)[0]
    label = class_names[predicted_class]
    confidence = np.max(predictions) * 100

    frame_with_boxes = detect_abnormalities(frame)

    cv2.putText(frame_with_boxes, f"Prediction: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame_with_boxes, f"Confidence: {confidence:.2f}%", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Diabetic Retinopathy Detection', frame_with_boxes)

    if cv2.getWindowProperty('Diabetic Retinopathy Detection', cv2.WND_PROP_VISIBLE) < 1:
        break
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
