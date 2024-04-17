import cv2
import os

# Load pre-trained Haar Cascade face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Directory containing images
image_folder = '../dataset/augmented/surprise'

# Count of images deleted
deleted_count = 0

# Maximum number of images to delete
max_deleted = 2567

for filename in os.listdir(image_folder):
    if deleted_count >= max_deleted:
        break
    
    # Check if file is an image
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        filepath = os.path.join(image_folder, filename)
        
        # Read image
        img = cv2.imread(filepath)
        
        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # If no face detected, delete the file
        if len(faces) == 0:
            os.remove(filepath)
            print(f"Deleted: {filepath}")
            deleted_count += 1

print(f"Total {deleted_count} images deleted.")
