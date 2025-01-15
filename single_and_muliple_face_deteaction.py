import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# read images
image = cv2.imread("images/single.webp") # single face detection
# image = cv2.imread('muliplefaces.jpg') # multiple face detection 

# convert to grayscale
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# gray_resized_image = cv2.resize(gray,(800,500))
# cv2.imshow("gray color images",gray_resized_image)

faces = face_cascade.detectMultiScale(gray)

# print(faces)

for X,Y,W,H in faces:
    cv2.rectangle(image,(X,Y),(X+W,Y+H),(255,0,0),2)

# resized_image = cv2.resize(image, (400,400))
# cv2.imshow('single image',resized_image)
cv2.imshow("single",image)

cv2.waitKey()
cv2.destroyAllWindows()