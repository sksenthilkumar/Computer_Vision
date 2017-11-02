from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import pandas


def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])

	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def compare_images(imageA, imageB, title):

    imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    imageA = cv2.resize(imageA,(300,300))
    imageB = cv2.resize(imageB,(300,300))
    # compute the mean squared error and structural similarity
	# index for the images
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)
    # setup the figure
    fig = plt.figure(title)
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap = plt.cm.gray)
    plt.axis("off")
    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap = plt.cm.gray)
    plt.axis("off")
	# show the images
    return m,s
    #plt.show()


images = [cv2.imread(file) for file in glob.glob("Data/*.jpg")]

Image_names = glob.glob("Data/*.jpg")

for i in range (len(Image_names)):
    Image_names[i]=Image_names[i].replace("Data/Label","")
    Image_names[i]=Image_names[i].replace(".jpg","")

size = len(images)
m_array = np.zeros((size,size),float)
s_array = np.zeros((size,size),float)
for i in range(len(images)):
    for j in range(len(images)):
        m,s = compare_images(images[i],images[j],"Hi")
        m_array[i][j]=int(m)
        s_array[i][j]=round (s,4)

for i,name in enumerate(Image_names):
    images[i] = cv2.resize(images[i],(300,300))
    cv2.imshow(name,images[i])


print ("MSI_Values:")
print (pandas.DataFrame(m_array, Image_names, Image_names))
print ("-------------------------------------------------")
print ("SSIM_Values:")
print (pandas.DataFrame(s_array, Image_names, Image_names))
cv2.waitKey(0)
