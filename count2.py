from skimage import io, filters
from scipy import ndimage
from skimage import measure
import matplotlib.pyplot as plt

im = io.imread("/home/sirshendu/Desktop/CS_783/Dataset/camera1/JPEGImages/00435.jpg", as_grey=True)
val = filters.threshold_otsu(im)
drops = ndimage.binary_fill_holes(im < val)
plt.imshow(drops, cmap='gray')
plt.show()

labels = measure.label(drops)
print(labels.max())