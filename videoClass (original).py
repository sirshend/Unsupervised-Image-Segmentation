import numpy as np
from skvideo.io import VideoCapture
import cv2
import skimage

def GetBackground(fileName):
    cap = VideoCapture(fileName)
    cap.open()
    bgImage = np.zeros((cap.height, cap.width, 3))
    numFrames = 0
    for i in range(400):
        retval, image = cap.read()

    while True:
        retval, image = cap.read()
        if not retval:
            break
        bgImage += image
        numFrames += 1    
        if (numFrames%50 == 0):
            print (numFrames)
        if (numFrames > 1200):
            break
    cap.release()
    bgImage = (bgImage/(numFrames*(1.0)))
    bgImage = bgImage.astype(np.uint8)
    return bgImage

def PlayVideo(fileName):
    '''
    Plays video in grayscale using opencv functions
    Press 'q' to stop in between
    returns None
    '''
    cap = VideoCapture(fileName)
    cap.open()
    for i in range(0,self.numFrames):
        gray = cv2.cvtColor(self.frames[i], cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

class video:
    def __init__(self):
        self.width = 0                  #Width of video frames, assuming all frames have the same resolution
        self.length = 0                   #Length of the frames 
        self.numFrames = 0                 #The total number of frames in the video
        self.frames = []                    #List of the frames (matrices) in order

    def VideoRead(self, fileName):
        """
        Reads the video from the specified fileName
        return value: None
        """
        '''
        Memory Intensive function
        '''
        '''
        cap = VideoCapture(fileName)
        cap.open()
        cnt = 0
        retval = True
        while retval:
            retval, image = cap.read()    
            if not retval:
                break
            self.frames.append(image)
            cnt += 1
            print (cnt)
        self.width = len(self.frames[-1])
        self.length = len(self.frames[-1][0])
        self.numFrames = len(self.frames)
        '''

    def PlayVideo(self):
        '''
        Plays video in grayscale using opencv functions
        Press 'q' to stop in between
        returns None
        '''
        for i in range(0,self.numFrames):
            gray = cv2.cvtColor(self.frames[i], cv2.COLOR_BGR2GRAY)
            cv2.imshow('frame',gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
