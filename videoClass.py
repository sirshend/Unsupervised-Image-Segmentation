import numpy as np
#from skvideo.io import VideoCapture
import cv2
import skimage

def GetBackground():
    image_dir="/home/sirshendu/Desktop/CS_783/Dataset/camera5/JPEGImages"
    file_names=[ os.path.join(image_dir,f) for f in os.listdir(image_dir) if f.endswith(".jpg")]
    #cap = VideoCapture(fileName)
    #cap.open()
    bgImage = np.zeros((622,830, 3))
    numFrames = 0
    for i in range(810):
        img=cv2.imread(file_names[i])
        bgImage+=img

    '''while True:
        retval, image = cap.read()
        if not retval:
            break
        bgImage += image
        numFrames += 1    
        if (numFrames%50 == 0):
            print (numFrames)
        if (numFrames > 1200):
            break'''
    #cap.release()
    bgImage = (bgImage/(810*(1.0)))
    bgImage = bgImage.astype(np.uint8)
    return bgImage
