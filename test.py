import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm  


model = tf.keras.models.load_model('diabetic_retinopathy_xception_model.h5')


base_dir = r"C:\Users\JOHN7\Documents\Visual Code\DRD\archive\organized_dataset"
train_dir = os.path.join(base_dir, "train")
validation_dir = os.path.join(base_dir, "validation")


class_names = ['no_DR', 'mild_DR', 'moderate_DR', 'severe_DR', 'proliferative_DR']


output_dir = os.path.join(base_dir, "sorted_predictions")
os.makedirs(output_dir, exist_ok=True)


for class_name in class_names:
    os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(299, 299))  
    img_array = img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0  
    return img_array


def process_images_from_directory(directory):
    for img_name in tqdm(os.listdir(directory), desc=f"Processing {directory}"):
        if img_name.lower().endswith(('.jpeg', '.jpg')): 
            img_path = os.path.join(directory, img_name)

           
            img_array = preprocess_image(img_path)

           
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction, axis=1)[0] 
            predicted_label = class_names[predicted_class]

          
            dest_path = os.path.join(output_dir, predicted_label, img_name)
            shutil.copy(img_path, dest_path)


process_images_from_directory(train_dir)
process_images_from_directory(validation_dir)

print("\n✅ All images have been predicted and sorted successfully!")
