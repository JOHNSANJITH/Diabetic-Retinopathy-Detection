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

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    preprocessed_frame = preprocess_image(frame)

    prediction = model.predict(preprocessed_frame)
    predicted_class = np.argmax(prediction, axis=1)
    label = class_names[predicted_class[0]]
    confidence = np.max(prediction) * 100

    cv2.putText(frame, f"Prediction: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"Confidence: {confidence:.2f}%", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Real-time Diabetic Retinopathy Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
