import cv2
import numpy as np

# Videodan okuma
cap = cv2.VideoCapture('video.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Gri tonlamaya dönüştür
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Kenar tespiti
    edges = cv2.Canny(gray, 50, 150)

    # Konturları bul
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Her bir kontur üzerinde döngü
    for contour in contours:
        # Eğer kontur yeterince büyükse
        if cv2.contourArea(contour) > 500:
            # Elipsi uygun şekilde uydur
            ellipse = cv2.fitEllipse(contour)
            
            # Elipsin merkezini ve ekseni uzunluklarını al
            (cx, cy), (major_axis, minor_axis), angle = ellipse
            
            # Şeklin türünü çıkar
            shape = ""
            if abs(major_axis - minor_axis) < 10:  # Yaklaşık olarak aynı olanlar daire olarak kabul edilir
                shape = "Daire"
            elif major_axis > minor_axis:  # Dikdörtgen
                shape = "Dikdörtgen"
            else:  # Elips
                shape = "Elips"
            
            # Şeklin türünü ekrana yazdır
            cv2.putText(frame, shape, (int(cx) - 50, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            

    # Sonuçları göster
    cv2.imshow('Frame', frame)

    # Çıkış için 'q' tuşuna bas
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Pencereyi kapat ve belleği serbest bırak
cap.release()
cv2.destroyAllWindows()
