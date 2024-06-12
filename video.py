import cv2
import numpy as np


cap = cv2.VideoCapture('video.mp4')

while(cap.isOpened()):
   
    ret, frame = cap.read()
    if not ret:
        break

   
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    
    edges = cv2.Canny(blur, 50, 150)

    
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        if len(approx) == 3:
            cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)  
            cv2.putText(frame, 'Triangle', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        elif len(approx) == 4:
            if 0.95 <= aspect_ratio <= 1.05:
                cv2.drawContours(frame, [approx], 0, (0, 0, 255), 2)  
                cv2.putText(frame, 'Square', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                cv2.drawContours(frame, [approx], 0, (255, 0, 0), 2)  
                cv2.putText(frame, 'Rectangle', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        elif len(approx) > 4:
            ellipse = cv2.fitEllipse(contour)
            cv2.ellipse(frame, ellipse, (0, 255, 255), 2) 
            cv2.putText(frame, 'Ellipse', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    
    cv2.imshow('Video', frame)

   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
