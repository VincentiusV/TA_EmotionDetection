import pandas as pd
import numpy as np
import os
import cv2

# Load the CSV file
df = pd.read_csv('../dataset/ckextended.csv')

# Remove the 'Usage' column
df.drop(columns=['Usage'], inplace=True)

# Define emotion mapping
emotion_mapping = {
    0: 'Anger',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happiness',
    4: 'Sadness',
    5: 'Surprise',
    6: 'Neutral',
    7: 'Contempt'
}

# Create emotion folders if they don't exist
for emotion in emotion_mapping.values():
    if not os.path.exists(emotion):
        os.makedirs(emotion)

# Transform and save images
for index, row in df.iterrows():
    emotion_label = row['emotion']
    pixel_data = np.fromstring(row['pixels'], dtype=int, sep=' ')
    
    # Reshape the pixel data to 48x48 grayscale image
    img = pixel_data.reshape(48, 48)
    
    # Save the image to the corresponding emotion folder
    emotion_folder = emotion_mapping.get(emotion_label)
    if emotion_folder:
        img_path = os.path.join(emotion_folder, f'image_{index}.png')
        cv2.imwrite(img_path, img)

print("Dataset transformation completed!")
