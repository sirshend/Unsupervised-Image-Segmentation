#from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
#import cv2
import sys
import numpy as np
from skimage import segmentation
import torch.nn.init


########################################################################################
#########################################################################################
##########################################################################################
##########################$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#############################
###########################################$$$$$$$$$$$$###############$$$$$$$$$###########



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
image_dir="/home/sirshendu/Desktop/CS_783/Dataset/camera4/JPEGImages"
file_names=[ os.path.join(image_dir,f) for f in os.listdir(image_dir) if f.endswith(".jpg")]
bgImage = np.zeros((622,830, 3))
numFrames = 0
for i in range(800):
    img=cv2.imread(file_names[i])
    bgImage+=img
bgImage = (bgImage/(800*(1.0)))
bgImage = bgImage.astype(np.uint8)
print(type(bgImage))
print(bgImage.shape) 
#return bgImage
#cv2.imshow(bgImage)
cv2.imwrite("~/Desktop/cam1000_background.jpg",bgImage)
cv2.imshow("see_cam1000_background ", bgImage)
cv2.waitKey()

graybg = cv2.cvtColor(bgImage, cv2.COLOR_BGR2GRAY)
#graybg = cv2.resize(graybg ,None, fx=0.25, fy=0.25)
cv2.imwrite("test_grey_shape.jpg",graybg)
io.imshow(graybg)
io.show()

face_cascade = cv2.CascadeClassifier('/home/sirshendu/Desktop/opencv-4.1.0/data/haarcascades/haarcascade_fullbody.xml')
plates=cv2.CascadeClassifier('/home/sirshendu/Desktop/opencv-4.1.0/data/haarcascades/haarcascade_licence_plate_rus_16stages.xml')
#33333333333333333333333333###############################################################################################################################################
###############################################################333333333333333333333333333333333333333333333333333333333333333############################################
#################################################################################################################################33333333333333333333333333333############
################################333333333333333333333333333333333333333333333#####################################3333333333333333333#################################33##
########################################################################3333333333333333333333######################333333333333#######################3333333333#########
#############################################3333333333333333###########################333333333333###################3333333333###############33333333#########3333#####
######333333333333333333#############################333333333#################333333333#################333333#################33333###########333333######333333333#####
##########################################################################################################################################################################
file__n=["/home/sirshendu/Desktop/CS_783/Dataset/camera4/JPEGImages/00353.jpg"]
for f in file__n:
    print("the file is: "+str(f))
    rcount=0
    frame=cv2.imread(f)
    #frame = cv2.resize(frame ,None, fx=0.25, fy=0.25)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)





    faces = plates.detectMultiScale(gray)
 
#print type(faces)
 
    if len(faces) == 0:
        print ("No faces found")
 
    else:
        #print faces
        #print faces.shape
        print ("Number of faces detected: " + str(faces.shape[0]) +str(','))
     
        for (x,y,w,h) in faces:
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),1)
     
        cv2.rectangle(image, ((0,image.shape[0] -25)),(270, image.shape[0]), (255,255,255), -1)
        cv2.putText(image, "Number of faces detected: " + str(faces.shape[0]), (0,image.shape[0] -10), cv2.FONT_HERSHEY_TRIPLEX, 0.5,  (0,0,0), 1)
 
        cv2.imshow('Image with people',image)
        cv2.waitKey(0)
        #cv2.destroyAllWindows()



    frame22=cv2.absdiff(bgImage,frame)
    cv2.imshow("Trial run BEFORE",frame22)
    #print(frame22.shape)
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################

    for m in range(622):
        for n in range(830):
            for o in range(3):
                if frame22[m][n][o] <=100:
                    frame22[m][n][o]=0


    cv2.imshow("Trial run AFTER",frame22)

    ##############################################################################
    ##############################################################################
    
    frameDelta = cv2.absdiff(graybg, gray)
    thresh = cv2.threshold(frameDelta, 100, 255, cv2.THRESH_BINARY)[1]
    #frame23 = cv2.threshold(frame, 78, 255, cv2.THRESH_BINARY)[1]




 
    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image

    thresh = cv2.GaussianBlur(thresh, (5, 5), 3)
    #frame23 = cv2.GaussianBlur(frame23, (5, 5), 3)


    cv2.imshow('OrigThresh', thresh)

    thresh = cv2.dilate(thresh, None, iterations = 4)

    cv2.imshow('DilateThresh', thresh)
    cv2.imwrite('test_dialte.jpg',thresh)
    #print(thresh.shape)
    #print(thresh)
    #break
    #thresh = cv2.GaussianBlur(thresh, (5, 5), 3)

    #cv2.imshow('BlurThresh', thresh)

    kernel = np.ones((5,5),np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)

    cv2.imshow('MorphThresh', thresh)

    tmpImg = thresh.copy()
    _,cnts, _ = cv2.findContours(tmpImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow('Cont', tmpImg)

    cv2.drawContours(gray, cnts, -1, (0,255,0), 3)
    cv2.imshow('DrawCont', gray)

    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < 2500:
            continue
        rcount=rcount+1
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Not Empty"
    
    # draw the text and timestamp on the frame
    '''cv2.putText(frame, "Gate Status: {}".format(text), (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)'''
    print("The number of objects is  :" +str(rcount))
    print('\n')
    # show the frame and record if the user presses a key
    cv2.imshow("Feed", frame)
    #cv2.imshow("Thresh", thresh)
    #cv2.imshow("Frame Delta", frameDelta)
    # key = cv2.waitKey(1) & 0xFF
 
    # # if the `q` key is pressed, break from the lop
    # if key == ord("q"):
    #     break
    # break






    
    

# cv2.waitKey(100000000)
#cv2.destroyAllWindows()
'''for i in range(156):
    for j in range(208):
        print(str(thresh[i][j])+" ")
    #print("")'''

#############################   helper code for image reconstruction experiment ########################################3




























##########################################################################################
##########################################################################################
##########################################################################################
###########################$$$$$$$$$$$$$$$$$$$$$$$$################$$$$$$$$$$$$###########
###$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#$$$$$$$$$$$$$$$$##



use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation')
parser.add_argument('--nChannel', metavar='N', default=100, type=int, 
                    help='number of channels')
parser.add_argument('--maxIter', metavar='T', default=1000, type=int, 
                    help='number of maximum iterations')
parser.add_argument('--minLabels', metavar='minL', default=4, type=int, 
                    help='minimum number of labels')
parser.add_argument('--lr', metavar='LR', default=0.1, type=float, 
                    help='learning rate')
parser.add_argument('--nConv', metavar='M', default=2, type=int, 
                    help='number of convolutional layers')
parser.add_argument('--num_superpixels', metavar='K', default=10000, type=int, 
                    help='number of superpixels')
parser.add_argument('--compactness', metavar='C', default=100, type=float, 
                    help='compactness of superpixels')
parser.add_argument('--visualize', metavar='1 or 0', default=1, type=int, 
                    help='visualization flag')
parser.add_argument('--input', metavar='FILENAME',
                    help='input image file name', required=True)
args = parser.parse_args()

# CNN model
class MyNet(nn.Module):
    def __init__(self,input_dim):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, args.nChannel, kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(args.nChannel)
        self.conv2 = []
        self.bn2 = []
        for i in range(args.nConv-1):
            self.conv2.append( nn.Conv2d(args.nChannel, args.nChannel, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(args.nChannel) )
        self.conv3 = nn.Conv2d(args.nChannel, args.nChannel, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(args.nChannel)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        for i in range(args.nConv-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x

# load image
im = cv2.imread(args.input)
im=thresh
data = torch.from_numpy( np.array([im.transpose( (2, 0, 1) ).astype('float32')/255.]) )
if use_cuda:
    data = data.cuda()
data = Variable(data)

# slic
labels = segmentation.slic(im, compactness=args.compactness, n_segments=args.num_superpixels)
labels = labels.reshape(im.shape[0]*im.shape[1])
u_labels = np.unique(labels)
l_inds = []
for i in range(len(u_labels)):
    l_inds.append( np.where( labels == u_labels[ i ] )[ 0 ] )

# train
model = MyNet( data.size(1) )
if use_cuda:
    model.cuda()
    for i in range(args.nConv-1):
        model.conv2[i].cuda()
        model.bn2[i].cuda()
model.train()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
label_colours = np.random.randint(255,size=(100,3))
for batch_idx in range(args.maxIter):
    # forwarding
    optimizer.zero_grad()
    output = model( data )[ 0 ]
    output = output.permute( 1, 2, 0 ).contiguous().view( -1, args.nChannel )
    ignore, target = torch.max( output, 1 )
    im_target = target.data.cpu().numpy()
    nLabels = len(np.unique(im_target))
    if args.visualize:
        im_target_rgb = np.array([label_colours[ c % 100 ] for c in im_target])
        im_target_rgb = im_target_rgb.reshape( im.shape ).astype( np.uint8 )
        cv2.imshow( "output", im_target_rgb )
        cv2.waitKey(10)

    # superpixel refinement
    # TODO: use Torch Variable instead of numpy for faster calculation
    for i in range(len(l_inds)):
        labels_per_sp = im_target[ l_inds[ i ] ]
        u_labels_per_sp = np.unique( labels_per_sp )
        hist = np.zeros( len(u_labels_per_sp) )
        for j in range(len(hist)):
            hist[ j ] = len( np.where( labels_per_sp == u_labels_per_sp[ j ] )[ 0 ] )
        im_target[ l_inds[ i ] ] = u_labels_per_sp[ np.argmax( hist ) ]
    target = torch.from_numpy( im_target )
    if use_cuda:
        target = target.cuda()
    target = Variable( target )
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()

    print (batch_idx, '/', args.maxIter, ':', nLabels, loss.data[0])
    if nLabels <= args.minLabels:
        print ("nLabels", nLabels, "reached minLabels", args.minLabels, ".")
        break

# save output image
if not args.visualize:
    output = model( data )[ 0 ]
    output = output.permute( 1, 2, 0 ).contiguous().view( -1, args.nChannel )
    ignore, target = torch.max( output, 1 )
    im_target = target.data.cpu().numpy()
    im_target_rgb = np.array([label_colours[ c % 100 ] for c in im_target])
    im_target_rgb = im_target_rgb.reshape( im.shape ).astype( np.uint8 )
cv2.imwrite( "output.png", im_target_rgb )