from keras.preprocessing.image import ImageDataGenerator
from skimage import io
import os

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# The base directory where the train data is stored
base_train_dir = '../dataset/fer2013plus/train'

# The base directory where the augmented images will be saved
base_augmented_dir = '../augmented_neutral/train'

# List of emotions
# emotions = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']

emotions = ['neutral']

# Iterate over each emotion
for emotion in emotions:
    # Path to the specific emotion directory
    emotion_dir = os.path.join(base_train_dir, emotion)
    
    # Ensure the augmented directory for the current emotion exists
    augmented_emotion_dir = os.path.join(base_augmented_dir, emotion)
    if not os.path.exists(augmented_emotion_dir):
        os.makedirs(augmented_emotion_dir)
    
    # List all images in the emotion directory
    image_files = os.listdir(emotion_dir)
    
    # Iterate over each image file
    for image_file in image_files:
        # Full path to the image
        image_path = os.path.join(emotion_dir, image_file)
        
        # Read the image for augmentation
        try:
            x = io.imread(image_path)
            x = x.reshape((1, ) + x.shape)
        except:
            continue  # If an image can't be read, skip it
        
        # Initialize counter
        i = 0
        # Generate and save the augmented images
        for batch in datagen.flow(x, batch_size=16, save_to_dir=augmented_emotion_dir, save_prefix='aug', save_format='png'):
            i += 1
            if i > 100:  # Change this to how many augmentations you want per image
                break  # Stop after generating the desired number of augmented images

        print(f"Augmented images for {image_file} saved in {augmented_emotion_dir}")


# from keras.preprocessing.image import ImageDataGenerator
# from skimage import io
# import os

# datagen = ImageDataGenerator(
#     rotation_range=15,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.1,
#     zoom_range=0.1,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

# # The base directory where the train data is stored
# base_train_dir = '/Users/vincentiusverel/Vincent/TugasAkhir/TA_EmotionDetection/dataset/fer2013plus/train'

# # The base directory where the augmented images will be saved
# base_augmented_dir = '/Users/vincentiusverel/Vincent/TugasAkhir/TA_EmotionDetection/dataset/augmented_split/train'

# # List of emotions
# emotions = ['neutral']

# # Iterate over each emotion
# for emotion in emotions:
#     # Path to the specific emotion directory
#     emotion_dir = os.path.join(base_train_dir, emotion)
    
#     # Ensure the augmented directory for the current emotion exists
#     augmented_emotion_dir = os.path.join(base_augmented_dir, emotion)
#     if not os.path.exists(augmented_emotion_dir):
#         os.makedirs(augmented_emotion_dir)
    
#     # List all images in the emotion directory
#     image_files = os.listdir(emotion_dir)
    
#     # Iterate over each image file
#     for image_file in image_files:
#         # Full path to the image
#         image_path = os.path.join(emotion_dir, image_file)
        
#         # Read the image for augmentation
#         try:
#             x = io.imread(image_path)
#             x = x.reshape((1, ) + x.shape)
#         except:
#             continue  # If an image can't be read, skip it
        
#         # Initialize counter for subdirectories
#         subdir_counter = 0
        
#         # Generate and save the augmented images
#         for batch in datagen.flow(x, batch_size=16, save_to_dir=augmented_emotion_dir, save_prefix='aug', save_format='png'):
#             subdir_counter += 1
            
#             # If we have generated 100 augmented images, create a new subdirectory
#             if subdir_counter % 10000 == 0:
#                 new_subdir = os.path.join(augmented_emotion_dir, f'subdir_{subdir_counter // 100}')
#                 if not os.path.exists(new_subdir):
#                     os.makedirs(new_subdir)
                
#             if subdir_counter > 10:  # Change this to how many augmentations you want per image
#                 break  # Stop after generating the desired number of augmented images

#         print(f"Augmented images for {image_file} saved in {augmented_emotion_dir}")
