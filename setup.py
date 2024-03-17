import os
import gdown
import zipfile
import shutil

# URL to your dataset on Google Drive
url = 'https://drive.google.com/file/d/1TJK1pmFRo8HTH2SW1N5U_MoqohyDl6xh/view?usp=sharing'

# Extract file ID from the URL
file_id = url.split('/')[-2]

# Direct download link template
direct_link = f'https://drive.google.com/uc?id={file_id}'

# Path where you want to save the downloaded zip file
zip_path = 'dataset.zip'

# Download the dataset from Google Drive
gdown.download(direct_link, zip_path, quiet=False)

# Create a folder to extract the dataset
extracted_dir = 'extracted_dataset'
os.makedirs(extracted_dir, exist_ok=True)

# Extract the dataset into the 'extracted_dataset' folder
try:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_dir)
    print("Dataset extracted successfully!")
except Exception as e:
    print("Error extracting dataset:", e)

# Remove the __MACOSX directory from the extracted folder
macosx_dir = os.path.join(extracted_dir, '__MACOSX')
if os.path.exists(macosx_dir):
    shutil.rmtree(macosx_dir)
    print("__MACOSX directory removed.")

# Remove the zip file after extraction
os.remove(zip_path)  # Remove the zip file from the current directory