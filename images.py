import cv2
import numpy as np

img0 = cv2.imread("Data/2.jpg",1) #RGB image
img2 = cv2.imread("Data/2.jpg",0) #grayscale image

img_edited = cv2.resize(img0,(600,600)) #resize the image

#saving all the images
cv2.imwrite("Data/image.jpg",img0)
cv2.imwrite("Data/image600x600.jpg",img_edited)
cv2.imwrite("Data/image_bw.jpg",img2)


cv2.imshow("img_bw01",img2)
cv2.imshow("img_rgb",img0)


cv2.waitKey(0)
cv2.destroyAllWindows()
