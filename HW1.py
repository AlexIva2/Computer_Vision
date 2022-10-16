import cv2
import numpy as np


img = cv2.imread('whiteballssample.jpg', cv2.IMREAD_UNCHANGED)

scale_percent = 50
img1 = cv2.resize(img, (int(img.shape[1] * scale_percent / 100), int(img.shape[0] * scale_percent / 100)))
img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

ret, img = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)
kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
erosion = cv2.erode(img, kernel_ellipse, iterations=1)
delation = cv2. dilate(erosion, kernel_ellipse, iterations=1)

circles = cv2.HoughCircles(delation, cv2.HOUGH_GRADIENT, 3, 20, param2=40, maxRadius=100)
circles = np.uint16(np.around(circles))
for (x, y, r) in circles[0, :]:
    cv2.circle(img1, (x, y), r, (0, 0, 0), 1)

print('mean:', round(circles[0, :, 2].mean()))
print('dispersion:', round(((circles[0, :, 2] - circles[0, :, 2].mean()) ** 2).mean()))

cv2.imshow('circles', img1)
cv2.waitKey()
