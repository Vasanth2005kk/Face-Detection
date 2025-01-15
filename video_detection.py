import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

video = cv2.VideoCapture("video/vid.mp4")

while True:
    success , frame = video.read()
    if success == True:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)
        print(faces)
        for X,Y,W,H in faces:
            cv2.rectangle(frame,(X,Y),(X+W,Y+H),(0,0,255),5)
        frame = cv2.resize(frame, (640, 480))  # Example: resizing to 640x480
        cv2.imshow("video",frame)
        key = cv2.waitKey(1)
        if key == 81 or key == 113:
            break
    else:
        print("Video is completed!")
        break