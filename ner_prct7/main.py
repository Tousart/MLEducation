import cv2

img = cv2.imread('mysite/media/image.png')
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('result', rgb)
cv2.waitKey(0)