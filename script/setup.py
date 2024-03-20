# import os
# import gdown
# import zipfile
# import shutil

# # URL to your dataset on Google Drive
# url = 'https://drive.google.com/file/d/1jRnIvaXiFL529uncdZwpx9p8ty1f7A1D/view?usp=sharing'

# # Extract file ID from the URL
# file_id = url.split('/')[-2]

# # Direct download link template
# direct_link = f'https://drive.google.com/uc?id={file_id}'

# # Path where you want to save the downloaded zip file
# zip_path = 'dataset.zip'

# # Download the dataset from Google Drive
# gdown.download(direct_link, zip_path, quiet=False)

# # Create a folder to extract the dataset
# extracted_dir = 'dataset'
# os.makedirs(extracted_dir, exist_ok=True)

# # Extract the dataset into the 'extracted_dataset' folder
# try:
#     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#         zip_ref.extractall(extracted_dir)
#     print("Dataset extracted successfully!")
# except Exception as e:
#     print("Error extracting dataset:", e)

# # Remove the __MACOSX directory from the extracted folder
# macosx_dir = os.path.join(extracted_dir, '__MACOSX')
# if os.path.exists(macosx_dir):
#     shutil.rmtree(macosx_dir)
#     print("__MACOSX directory removed.")

# # Remove the zip file after extraction
# os.remove(zip_path)  # Remove the zip file from the current directory

import os
import gdown
import zipfile
import shutil

def download_dataset(url, zip_path):
    try:
        gdown.download(url, zip_path, quiet=False)
        print("Dataset downloaded successfully!")
    except Exception as e:
        print("Error downloading dataset:", e)

def extract_dataset(zip_path, root_folder):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(root_folder)
        print("Dataset extracted successfully!")
    except Exception as e:
        print("Error extracting dataset:", e)

def remove_macosx_dir(root_folder):
    macosx_dir = os.path.join(root_folder, '__MACOSX')
    if os.path.exists(macosx_dir):
        shutil.rmtree(macosx_dir)
        print("__MACOSX directory removed.")

def remove_zip_file(zip_path):
    os.remove(zip_path)
    print("Zip file removed.")

def download_and_extract_dataset(url, root_folder):
    zip_path = os.path.join(root_folder, 'dataset.zip')
    download_dataset(url, zip_path)
    extract_dataset(zip_path, root_folder)
    remove_macosx_dir(root_folder)
    remove_zip_file(zip_path)

if __name__ == "__main__":
    # URL to your dataset on Google Drive
    url = 'https://drive.google.com/uc?id=1jRnIvaXiFL529uncdZwpx9p8ty1f7A1D'

    # Root folder where you want to extract the dataset
    root_folder = '../'

    # Download and extract the dataset
    download_and_extract_dataset(url, root_folder)