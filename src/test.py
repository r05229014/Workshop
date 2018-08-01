import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('../Manhattan/Manhattan00000.png', cv2.IMREAD_GRAYSCALE)

print(img.shape)
print(img)


#plt.imshow(img, cmap = 'gray')
#plt.show()
