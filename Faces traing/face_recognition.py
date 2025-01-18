# pylint:disable=no-member

import numpy as np
import cv2 as cv
import os

# Load Haar Cascade file
haar_cascade_path = f"{os.path.abspath(os.getcwd())}/haarcascade_frontalface_default.xml"
haar_cascade = cv.CascadeClassifier(haar_cascade_path)

# Validate if Haar Cascade is loaded successfully
if haar_cascade.empty():
    raise FileNotFoundError(f"Haar Cascade file not found or failed to load from {haar_cascade_path}")

# Define the list of people (labels)
people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']

# Load the trained model
trained_model_path = "/".join(f'{os.path.abspath(__file__)}'.split("/")[:-1:])+"/face_trained.yml"

# print(trained_model_path)
if not os.path.exists(trained_model_path):
    raise FileNotFoundError(f"Trained model file not found: {trained_model_path}")

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read(trained_model_path)

# Specify the path to the image for validation
image_path =f"{os.path.abspath(os.getcwd())}/Resources/Faces/val/elton_john/1.jpg"
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file not found: {image_path}")

# Load the image
img = cv.imread(image_path)
if img is None:
    raise ValueError(f"Failed to load image from {image_path}")

# Convert the image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Person', gray)

# Detect the face in the image
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

# Process detected faces
for (x, y, w, h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+w]

    # Predict the label and confidence
    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence:.2f}')

    # Display the predicted label and rectangle around the face
    cv.putText(img, f'{people[label]} ({confidence:.2f})', (x, y - 10), cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), thickness=2)
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=1)

# Show the image with detected faces and predictions
# resized_image = cv.resize(img, (800,500))
cv.imshow('Detected Face', img)


# Wait for a key press before exiting
cv.waitKey(0)
cv.destroyAllWindows()
