from videoClass import *
import skimage.io as io
import pickle
import datetime
import time
import cv2
import os
import skimage
from skimage import data
import random
import numpy as np 
import sklearn as sk
from sklearn import svm
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier('/home/sirshendu/Desktop/opencv-4.1.0/data/haarcascades/haarcascade_frontalface_default.xml')
 
image = cv2.imread('/home/sirshendu/Desktop/CS_783/Dataset/camera1/JPEGImages/00756.jpg')
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
image_dir="/home/sirshendu/Desktop/CS_783/Dataset/camera5/JPEGImages"
file_names=[ os.path.join(image_dir,f) for f in os.listdir(image_dir) if f.endswith(".jpg")]
for f in file_names:
    img=cv2.imread(f)
    print(img.shape)
    print("\n")
