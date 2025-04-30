Diabetic Retinopathy Detection using Deep Learning


This project applies deep learning to detect diabetic retinopathy, a complication of diabetes that affects the retina and can lead to blindness if not diagnosed in time. Using a CNN trained on thousands of labeled retinal images. 
The model can classify the severity of the disease into five categories: 
• No DR 
• Mild
• Moderate 
• Severe 
• Proliferative DR 
The use of such a model not only accelerates the screening process but also increases accessibility in regions with limited access to ophthalmologists. By leveraging image preprocessing, data augmentation, and advanced architectures (like ResNet or VGG), the model learns to distinguish subtle visual features indicative of disease progression. This kind of DL-powered system has the potential to be integrated into real-time clinical tools, telemedicine platforms, or mobile diagnostic apps—supporting better healthcare outcomes through early, accurate, and scalable detection.



Model Overview
• Model: CNN (e.g., ResNet50, VGG16, or custom architecture)

• Framework: TensorFlow / Keras / PyTorch

• Classes: No DR, Mild, Moderate, Severe, Proliferative DR



Dataset
• Dataset Used: Kaggle Diabetic Retinopathy Resized Dataset

• Source: https://www.kaggle.com/datasets/tanlikesmath/diabetic-retinopathy-resized

• Preprocessing: Images were resized, normalized, and augmented.



Features
• Image preprocessing and augmentation

• Training and validation split

• Evaluation using accuracy, confusion matrix, and AUC

• Model saving and loading



Future Work
• Integration with a web application

• Improved model explainability (e.g., Grad-CAM visualization)

• Deploying the model as an API
