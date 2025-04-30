import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split


dataset_path = r"C:\Users\JOHN7\Documents\Visual Code\DRD\archive\organized_dataset"
train_folder = os.path.join(dataset_path, "train")
val_folder = os.path.join(dataset_path, "validation")


class_names = ['no_DR', 'mild_DR', 'moderate_DR', 'severe_DR', 'proliferative_DR']
for class_name in class_names:
    os.makedirs(os.path.join(val_folder, class_name), exist_ok=True)


for class_name in class_names:
    class_train_folder = os.path.join(train_folder, class_name)
    class_val_folder = os.path.join(val_folder, class_name)

    images = os.listdir(class_train_folder)
    train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

    for img in val_images:
        src_path = os.path.join(class_train_folder, img)
        dest_path = os.path.join(class_val_folder, img)
        shutil.move(src_path, dest_path)

print("✅ Validation folder created successfully!")
