import numpy as np
import cv2
from keras.models import load_model

# Load the emotion detection model
model = load_model('../training/100_custom2_augmented/100_custom2_augmented.h5')

# Define the emotions
emotion_labels = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']

def predict_emotion(image_path):
    # Load and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (48, 48))
    img = np.reshape(img, (1, 48, 48, 1))
    img = img / 255.0  # Normalize

    # Predict emotion
    predictions = model.predict(img)
    
    # Get the index of the highest probability
    predicted_index = np.argmax(predictions)
    
    # Get the predicted emotion label
    predicted_emotion = emotion_labels[predicted_index]
    
    return predicted_emotion

# Example usage
image_path = 'example_image.jpg'
predicted_emotion = predict_emotion('../dataset/CK+/Contempt/image_671.png')
print("Predicted Emotion:", predicted_emotion)
