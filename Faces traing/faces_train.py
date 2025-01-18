# pylint:disable=no-member

import os
from pathlib import Path
import cv2 as cv
import numpy as np

# Define people and training directory
people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
DIR = os.path.abspath('./Resources/Faces/images_Train')  # Relative to script location

# Load Haar Cascade
CASCADE_PATH = os.path.abspath('./haarcascade_frontalface_default.xml')
haar_cascade = cv.CascadeClassifier(CASCADE_PATH)

# Validate Haar Cascade loading
if haar_cascade.empty():
    raise FileNotFoundError(f"Haar Cascade XML file not found or failed to load from {CASCADE_PATH}")

# Initialize feature and label lists
features = []
labels = []

def create_train():
    """Prepare training data by detecting faces in images and extracting regions of interest."""
    if not os.path.exists(DIR):
        raise FileNotFoundError(f"Training directory {DIR} does not exist.")

    for person in people:
        path = Path(DIR) / person
        label = people.index(person)

        if not path.exists():
            print(f"Directory {path} not found. Skipping {person}.")
            continue

        for img in os.listdir(path):
            if not img.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"Skipping non-image file: {img}")
                continue

            img_path = path / img
            img_array = cv.imread(str(img_path))
            if img_array is None:
                print(f"Failed to load {img_path}. Skipping.")
                continue

            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            faces_rect = haar_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                faces_roi = cv.resize(faces_roi, (100, 100))  # Normalize size
                # Ensure correct data type
                features.append(faces_roi.astype(np.uint8))
                labels.append(label)

    print(f"Processed {len(features)} faces from training data.")

# Run training data preparation
try:
    create_train()
except FileNotFoundError as e:
    print(e)
    exit()

if len(features) == 0 or len(labels) == 0:
    print("No valid training data found. Exiting.")
    exit()

# Convert labels to NumPy array (features are already a list of NumPy arrays)
labels = np.array(labels, dtype=np.int32)

# Validate if OpenCV's LBPH Face Recognizer is available
if not hasattr(cv.face, 'LBPHFaceRecognizer_create'):
    raise AttributeError("OpenCV installation does not include the `cv2.face` module. Ensure you have the correct OpenCV version installed.")

# Initialize and train the LBPH face recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features, labels)

# Save the trained recognizer and data
face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)

print("Training completed and model saved successfully.")
