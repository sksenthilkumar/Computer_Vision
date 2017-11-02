import cv2
import glob

images = [cv2.imread(file) for file in glob.glob("Data/*.jpg")]
print (images)

#cv2.imshow("all",images[0])
for img in images:
    cv2.imshow("all",img)
    cv2.waitKey(550)
cv2.waitKey(0)
cv2.destroyAllWindows()
