import os
import shutil
import pandas as pd


dataset_path = r"C:\Users\JOHN7\Documents\Visual Code\DRD\archive"
train_images_path = os.path.join(dataset_path, "resized_train", "resized_train") 
labels_csv = os.path.join(dataset_path, "trainLabels.csv")


class_names = ['no_DR', 'mild_DR', 'moderate_DR', 'severe_DR', 'proliferative_DR']


train_output = os.path.join(dataset_path, "organized_dataset", "train")
val_output = os.path.join(dataset_path, "organized_dataset", "validation")


df = pd.read_csv(labels_csv)


def move_images(df, src_folder, dest_folder):
    moved_count = 0
    missing_count = 0

    for _, row in df.iterrows():
        img_name = row['image'] + ".jpeg"  
        class_index = row['level']  
        class_folder = class_names[class_index]

        src_path = os.path.join(src_folder, img_name)
        dest_path = os.path.join(dest_folder, class_folder, img_name)

        if os.path.exists(src_path):
            shutil.move(src_path, dest_path)
            moved_count += 1
        else:
            missing_count += 1
            print(f"⚠️ Missing: {src_path}")  

    print(f"✅ Moved {moved_count} images successfully!")
    print(f"⚠️ {missing_count} images not found.")

move_images(df, train_images_path, train_output)
