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
import sys
from sys import argv

# print(argv[1])
# file_names = []
# for i in range(1,6):
#     camera = "camera" + str(i)
num = str(argv[1]) + "_"
camera = "camera" + str(argv[1])
image_dir = "/home/sirshendu/Desktop/Dataset/"+camera+"/JPEGImages"
file_names = [ os.path.join(image_dir,f) for f in os.listdir(image_dir) if f.endswith(".jpg")]

img_count = len(file_names)
print("Total images in "+ camera+" are:",img_count)
print("--------------------------------")
bgImage = np.zeros((622,830, 3))
numFrames = 0

for i in file_names:
    img = cv2.imread(i)
    bgImage += img

bgImage = (bgImage/(img_count*(1.0)))
bgImage = bgImage.astype(np.uint8)
# print(type(bgImage))
# print(bgImage.shape) 
#return bgImage
#cv2.imshow(bgImage)
cv2.imwrite("/home/sirshendu/Desktop/output/background_image_coloured_"+num+".jpg",bgImage)
# cv2.imshow("averaged_backgroud", bgImage)
# cv2.waitKey(0)

graybg = cv2.cvtColor(bgImage, cv2.COLOR_BGR2GRAY)
#graybg = cv2.resize(graybg ,None, fx=0.25, fy=0.25)
cv2.imwrite("/home/sirshendu/Desktop/output/background_image_gray_"+num+".jpg",graybg)
# io.imshow(graybg)
# io.show()


face_cascade = cv2.CascadeClassifier('/home/sirshendu/opencv-3.4/data/haarcascades/haarcascade_fullbody.xml')
plates=cv2.CascadeClassifier('/home/sirshendu/opencv-3.4/data/haarcascades/haarcascade_licence_plate_rus_16stages.xml')
#33333333333333333333333333###############################################################################################################################################
###############################################################333333333333333333333333333333333333333333333333333333333333333############################################
#################################################################################################################################33333333333333333333333333333############
################################333333333333333333333333333333333333333333333#####################################3333333333333333333#################################33##
########################################################################3333333333333333333333######################333333333333#######################3333333333#########
#############################################3333333333333333###########################333333333333###################3333333333###############33333333#########3333#####
######333333333333333333#############################333333333#################333333333#################333333#################33333###########333333######333333333#####
##########################################################################################################################################################################

cn = 0
for f in file_names:
    cn += 1
    # f = '/home/sirshendu/Desktop/Dataset/camera4/JPEGImages/00211.jpg'
    fil = f.strip().split('/')
    fil = fil[len(fil) - 1]
    # print(fil)
    print("the file is: "+ camera + " => "+str(fil))
    rcount = 0
    frame = cv2.imread(f)
    image = frame.copy()
    # print(f)
    #frame = cv2.resize(frame ,None, fx=0.25, fy=0.25)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = plates.detectMultiScale(gray)
 
    # if len(faces) == 0:
    #     print ("No faces found")

    # else:
    #     #print faces
    #     #print faces.shape
    #     print("Number of faces detected in " + str(f)+ ":" + str(faces.shape[0]) +str(','))
     
    #     for (x,y,w,h) in faces:
    #         cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),1)
     
    #     cv2.rectangle(image, ((0,image.shape[0] -25)),(270, image.shape[0]), (255,255,255), -1)
    #     cv2.putText(image, "Number of faces detected: " + str(faces.shape[0]), (0,image.shape[0] -10), cv2.FONT_HERSHEY_TRIPLEX, 0.5,  (0,0,0), 1)
 
    #     # cv2.imshow('Image with people',image)
    #     cv2.imwrite("output/Image_with_people_"+ str(fil),image)
    #     # cv2.waitKey(0)
    #     #cv2.destroyAllWindows()



    frame22=cv2.absdiff(bgImage,frame)
    # cv2.imshow("Trial run BEFORE",frame22)
    cv2.imwrite("output/Trial_run_BEFORE_"+str(num+fil), frame22)
    # cv2.waitKey(0)
    # print(frame22.shape)
    ############################################################################### 
    ###############################################################################
    ###############################################################################
    ###############################################################################

    new_thresh = 100
    for m in range(622):
        for n in range(830):
            for o in range(3):
                if frame22[m][n][o] <= new_thresh:
                    frame22[m][n][o]=0


    # cv2.imshow("Trial run AFTER",frame22)
    cv2.imwrite('output/Trial_run_AFTER_'+ str(num+fil),frame22)
    frame23 = cv2.GaussianBlur(frame22, (5, 5), 3)    
    cv2.imwrite('output/Trial_run_AFTER_blurred_'+str(num+fil),frame23)

    ##############################################################################
    ##############################################################################
    
    frameDelta = cv2.absdiff(graybg, gray)
    thresh = cv2.threshold(frameDelta, 100, 255, cv2.THRESH_BINARY)[1]
    thresh3 = cv2.GaussianBlur(frameDelta, (5, 5), 3)
    threshhhh=cv2.threshold(thresh3,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    
    # cv2.imshow("Otsu",threshhhh)
    cv2.imwrite('output/Otsu_'+ str(num+fil),threshhhh)
    #frame23 = cv2.threshold(frame, 78, 255, cv2.THRESH_BINARY)[1]




 
    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image

    thresh = cv2.GaussianBlur(thresh, (5, 5), 3)
    #frame23 = cv2.GaussianBlur(frame23, (5, 5), 3)


    # cv2.imshow('OrigThresh', thresh)
    cv2.imwrite('output/OrigThresh_'+ str(num+fil) ,thresh)

    thresh = cv2.dilate(thresh, None, iterations = 4)
    threshhhh = cv2.dilate(threshhhh, None, iterations = 4)

    # cv2.imshow('DilateThresh', thresh)
    cv2.imwrite('output/DilateThresh_'+str(num+fil) ,thresh)
    # cv2.imshow('Otsu thresh',threshhhh)
    cv2.imwrite('output/Otsu_thresh_'+str(num+fil) ,threshhhh)

    #print(thresh.shape)
    #print(thresh)
    #break
    #thresh = cv2.GaussianBlur(thresh, (5, 5), 3)

    #cv2.imshow('BlurThresh', thresh)

    kernel = np.ones((5,5),np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)
    threshhhh = cv2.morphologyEx(threshhhh, cv2.MORPH_OPEN, kernel, iterations = 2)
    
    # cv2.imshow('MorphThresh', thresh)
    cv2.imwrite('output/MorphThresh_'+str(num+fil) ,thresh)

    # cv2.imshow('otsu mprpho thresh',threshhhh)
    cv2.imwrite('output/otsu_mprpho_thresh_'+str(num+fil) ,threshhhh)

    threshhhh_c=threshhhh.copy()
    tmpImg = thresh.copy()
    _,cnts, _ = cv2.findContours(tmpImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _,cnts22, _ = cv2.findContours(threshhhh_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    
    
    # cv2.imshow('Cont', tmpImg)
    cv2.imwrite('output/Cont_'+str(num+fil) ,tmpImg)
    
    cv2.drawContours(gray, cnts, -1, (0,255,0), 3)
    
    # cv2.imshow('DrawCont', gray)
    cv2.imwrite('output/DrawCont_'+str(num+fil) ,gray)


    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < 2500:
            continue
        rcount = rcount + 1
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Not Empty"
    cv2.imwrite("output/frame_"+str(num+fil), frame)
    # draw the text and timestamp on the frame
    '''cv2.putText(frame, "Gate Status: {}".format(text), (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)'''
    print("The number of objects is/are ==> " +str(rcount))
    print('\n')
    # show the frame and record if the user presses a key
    # cv2.imshow("Feed", frame)
    cv2.imwrite('output/Feed_'+str(num+fil) ,tmpImg)

    # cv2.imshow("Thresh", thresh)
    # cv2.imshow("Frame Delta", frameDelta)
    # key = cv2.waitKey(1) & 0xFF
 
    # if the `q` key is pressed, break from the lop
    # if key == ord("q"):
        # break
    if cn == 25:
        break



# cv2.waitKey(0)
#cv2.destroyAllWindows()
'''for i in range(156):
    for j in range(208):
        print(str(thresh[i][j])+" ")
    #print("")'''

#############################   helper code for image reconstruction experiment ########################################3
