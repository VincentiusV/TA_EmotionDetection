## BUAT Pre-Trained

# import os
# import cv2
# import numpy as np
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Function to load and preprocess CK+ dataset
# def load_ck_plus_data(data_dir, img_size):
#     images = []
#     labels = []
#     for label_dir in os.listdir(data_dir):
#         label_path = os.path.join(data_dir, label_dir)
#         if os.path.isdir(label_path):
#             for img_name in os.listdir(label_path):
#                 img_path = os.path.join(label_path, img_name)
#                 if img_path.endswith('.png'):
#                     img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#                     img = cv2.resize(img, (img_size, img_size))
#                     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
#                     images.append(img)
#                     labels.append(label_dir)
#     images = np.array(images)
#     labels = np.array(labels)
#     return images, labels

# # Assuming the CK+ dataset is stored in "ck_plus_data" directory
# data_dir = '../dataset/CK+'
# img_size = 48  # assuming the model was trained on 48x48 images
# X_ck, y_ck = load_ck_plus_data(data_dir, img_size)

# # Normalize the images
# X_ck = X_ck / 255.0
# # X_ck = X_ck.reshape(-1, img_size, img_size, 1)

# # Define the label mapping and encode the labels
# label_mapping = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
# le = LabelEncoder()
# le.fit(label_mapping)
# y_ck_encoded = le.transform(y_ck)

# # Load the model trained on FERPlus
# # model = load_model('../training/100_custom2_masked/100_custom2_masked.h5')
# model = load_model('../training/100_vgg16_masked/100_vgg16_masked.h5')

# # Predict the labels for CK+ dataset
# y_pred_proba = model.predict(X_ck)
# y_pred = np.argmax(y_pred_proba, axis=1)

# # Calculate accuracy
# accuracy = accuracy_score(y_ck_encoded, y_pred)
# print(f'Accuracy: {accuracy:.2f}')

# # Calculate F1 score
# f1 = f1_score(y_ck_encoded, y_pred, average='weighted')
# print(f'F1 Score: {f1:.2f}')

# # Calculate confusion matrix
# conf_matrix = confusion_matrix(y_ck_encoded, y_pred, labels=range(len(label_mapping)))

# print('Confusion Matrix:')
# print(conf_matrix)

# # Plot confusion matrix
# plt.figure(figsize=(10, 8))
# sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=label_mapping, yticklabels=label_mapping, cmap='Blues')
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.title('Confusion Matrix')
# plt.show()


## BUAT CNN

import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load and preprocess CK+ dataset
def load_ck_plus_data(data_dir, img_size):
    images = []
    labels = []
    for label_dir in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label_dir)
        if os.path.isdir(label_path):
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                if img_path.endswith('.png'):
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (img_size, img_size))
                    images.append(img)
                    labels.append(label_dir)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

# Assuming the CK+ dataset is stored in "ck_plus_data" directory
data_dir = '../dataset/CK+'
img_size = 48  # assuming the model was trained on 48x48 images
X_ck, y_ck = load_ck_plus_data(data_dir, img_size)

# Normalize the images
X_ck = X_ck / 255.0
X_ck = X_ck.reshape(-1, img_size, img_size, 1)

# Define the label mapping and encode the labels
label_mapping = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
le = LabelEncoder()
le.fit(label_mapping)
y_ck_encoded = le.transform(y_ck)

# Load the model trained on FERPlus
model_path = '../training/100_custom2_masked/100_custom2_masked.h5'
model = load_model(model_path)

# Predict the labels for CK+ dataset
y_pred_proba = model.predict(X_ck)
y_pred = np.argmax(y_pred_proba, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_ck_encoded, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Calculate F1 score
f1 = f1_score(y_ck_encoded, y_pred, average='weighted')
print(f'F1 Score: {f1:.2f}')

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_ck_encoded, y_pred, labels=range(len(label_mapping)))

print('Confusion Matrix:')
print(conf_matrix)

# Plot confusion matrix with a brighter color map
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=label_mapping, yticklabels=label_mapping, cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()