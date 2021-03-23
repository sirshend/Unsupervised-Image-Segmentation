import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('/home/sirshendu/Desktop/CS_783/Dataset/camera1/JPEGImages/00001.jpg')
print(img)

print(img.shape)
b,g,r = cv2.split(img)           # get b,g,r
rgb_img = cv2.merge([r,r,r]) 
print(rgb_img.shape)    # switch it to rgb

# Denoising
dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

b,g,r = cv2.split(dst)           # get b,g,r
rgb_dst = cv2.merge([g,g,g])     # switch it to rgb
print(rgb_dst.shape)
plt.subplot(211),plt.imshow(rgb_img)
plt.subplot(212),plt.imshow(rgb_dst)
plt.show()