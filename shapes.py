import cv2
import numpy as np


image = cv2.imread('shapes.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)


contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
    if len(approx) == 3:
        cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)  
        x, y, w, h = cv2.boundingRect(contour)
        cv2.putText(image, 'ucgen', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    elif len(approx) == 4:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        if 0.95 <= aspect_ratio <= 1.05:
            cv2.drawContours(image, [approx], 0, (0, 0, 255), 2)  
            cv2.putText(image, 'Kare', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            cv2.drawContours(image, [approx], 0, (255, 0, 0), 2) 
            cv2.putText(image, 'Dikdortgen', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
