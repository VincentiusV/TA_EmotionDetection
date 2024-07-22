import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

def load_model(model_path):
    # Load the pre-trained model
    model = tf.keras.models.load_model(model_path)
    return model

def preprocess_image(img_path, target_size):
    # Load the image
    img = Image.open(img_path)
    # Convert the image to grayscale
    img = img.convert('L')
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to an array
    img_array = image.img_to_array(img)
    # Expand dimensions to match the model input
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize the image array
    img_array = img_array / 255.0
    return img_array

def predict_image(model, img_array):
    # Predict the label of the image
    predictions = model.predict(img_array)
    # Get the label with the highest score
    predicted_label = np.argmax(predictions, axis=1)[0]
    # Get the confidence of the prediction
    confidence = np.max(predictions, axis=1)[0]
    return predicted_label, confidence

def main():
    model_path = '/Users/vincentiusverel/Vincent/TugasAkhir/TA_EmotionDetection/training/100_custom2_masked/100_custom2_masked.h5'  # Replace with the path to your model
    img_path = '/Users/vincentiusverel/Vincent/TugasAkhir/TA_EmotionDetection/dataset/testing_image/anger.jpeg'   # Replace with the path to your image
    target_size = (48, 48)  # Replace with your model's input size
    
    # Define the label mapping
    label_mapping = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

    # Load the model
    model = load_model(model_path)
    # Preprocess the image
    img_array = preprocess_image(img_path, target_size)
    # Predict the label
        # Predict the label
    predicted_label_idx, confidence = predict_image(model, img_array)
    # Map the predicted label index to the label
    predicted_label = label_mapping[predicted_label_idx]
    

    
    # Display the result
    print(f'Predicted Label: {predicted_label}')
    print(f'Confidence: {confidence:.2f}')

if __name__ == '__main__':
    main()
