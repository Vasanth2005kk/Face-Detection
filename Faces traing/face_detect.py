# pylint:disable=no-member

import cv2 as cv
import os

# Set up the image path
image_location = os.path.abspath(os.getcwd() + '/Resources/Photos/group 1.jpg')

# Load the image
img = cv.imread(image_location)

# Convert the image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray People', gray)

# Load the Haar Cascade file
cascade_path = f'{os.path.abspath(os.getcwd())}/haarcascade_frontalface_default.xml'
haar_cascade = cv.CascadeClassifier(cascade_path)

# Validate if Haar Cascade is loaded successfully
if haar_cascade.empty():
    raise FileNotFoundError(f"Haar Cascade file not found or failed to load from {cascade_path}")

# Detect faces in the image
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)
print(f'Number of faces found = {len(faces_rect)}')

# Draw rectangles around detected faces
for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

# Show the image with detected faces
cv.imshow('Detected Faces', img)

cv.waitKey(0)
