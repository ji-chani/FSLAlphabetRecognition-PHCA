# **Filipino Sign Language Alphabet Recognition using Persistent Homology Classification Algorithm**

## **Authors:**

Cristian B. Jetomo and Mark Lexter D. De Lara

## **Abstract:**
Deaf or hearing-impaired individuals have been facing problems in communicating with the normal hearing population. To cope with this communication gap, numerous sign languages have been developed, one of which is the Filipino Sign Language (FSL). Despite FSL being declared as the national sign language of the Philippines, most Filipinos still do not understand the language. Hence, machine learning techniques are leveraged to automate the interpretation process of signed gestures and the field of sign language recognition is developed. This paper extends this field by utilizing computational topology-based methods in performing Filipino sign language recognition (FSLR). Specifically, it aims to utilize Persistent Homology Classification Algorithm (PHCA) in classifying or interpreting static alphabet signed using FSL. The performance of PHCA is evaluated in comparison with widely used classifiers. Validation runs shows that PHCA performed at par with the other classifiers considered.

## Instructions for Replication

### Data Preparation
- Download all codes and save in a directory/folder
- Download the dataset from https://www.kaggle.com/datasets/japorton/fsl-dataset?resource=download (filename: _archive.zip_) and save in the same directory
- Extract the folder in the zip file (foldername: _Collated_) and move in the directory
- Run the [rename_dataset.py](BreadcrumbsFSLAlphabetRecognition-PHCA/rename_dataset.py) file to rename the dataset filenames. You should expect a new folder named _FSL_images_ containing subfolders with images
- Compress the new folder into a zip file (filename: _FSL_images_static.zip_)

### Main Implementation
- Create a virtual environment and install all dependencies from [requirements.txt](BreadcrumbsFSLAlphabetRecognition-PHCA/requirements.txt) file (**Important Note**: Make sure to use Python version (>3.7, <=3.10.10) so that the _ripser_ package will work)
- Run the [main.py](BreadcrumbsFSLAlphabetRecognition-PHCA/main.py) file
- View the results (shown as images) in the directory
