import cv2
import numpy as np

img = cv2.imread("Data/image.jpg")

height, width = img.shape[0:2]
binary = np.zeros([height,width,1],'uint8')
thresh = 85

imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

x = 50 #contrast value

imghsv[:,:,2] = [[max(pixel - x, 0) if pixel < 200 else min(pixel + x, 255) for pixel in row] for row in imghsv[:,:,2]]
img_contrast = cv2.cvtColor(imghsv, cv2.COLOR_HSV2BGR)


cv2.imshow('contrast', img_contrast)
cv2.imshow("image",img)


cv2.waitKey(0)
cv2.destroyAllWindows()
