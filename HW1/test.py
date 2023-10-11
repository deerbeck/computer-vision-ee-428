import numpy as np
import cv2

def mse(img1, img2):
   h, w = img1.shape
   diff = cv2.subtract(img1, img2)
   err = np.sum(diff**2)
   mse = err/(float(h*w))
   return mse

filenames = ["L1004220.JPG", "L1004235.JPG", "L1004432.JPG"]
for filename in filenames:
    # read image
    img1 = cv2.imread(filename)
    img2 = cv2.imread("solution_images/"+filename)
    # extract color channels
    (B1, G1, R1) = cv2.split(img1)
    (B2, G2, R2) = cv2.split(img2)
    # get grayscale
    img1_g = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_g = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # display errors
    print()
    print(filename)
    print("TOTAL", mse(img1_g, img2_g),"%")
    print("R", mse(R1, R2),"%")
    print("G", mse(G1, G2),"%")
    print("B", mse(B1, B2),"%")
    