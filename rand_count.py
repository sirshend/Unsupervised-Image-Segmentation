import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('/home/sirshendu/Desktop/opencv-4.1.0/data/haarcascades/haarcascade_fullbody.xml')
 
image = cv2.imread('/home/sirshendu/Desktop/CS_783/Dataset/camera1/JPEGImages/00315.jpg')
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
faces = face_cascade.detectMultiScale(grayImage)
 
#print type(faces)
 
if len(faces) == 0:
    print ("No faces found")
 
else:
    #print faces
    #print faces.shape
    print ("Number of faces detected: " + str(faces.shape[0]))
 
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),1)
 
    cv2.rectangle(image, ((0,image.shape[0] -25)),(270, image.shape[0]), (255,255,255), -1)
    cv2.putText(image, "Number of faces detected: " + str(faces.shape[0]), (0,image.shape[0] -10), cv2.FONT_HERSHEY_TRIPLEX, 0.5,  (0,0,0), 1)
 
    cv2.imshow('Image with people',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()