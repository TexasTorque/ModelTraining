from ultralytics import YOLO
import cv2

cap = cv2.VideoCapture(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

model = YOLO('runs/detect/train8/weights/best.pt')

while True:
    ret, frame = cap.read()

    results = model(frame)

    cv2.imshow("frame", results[0].plot())

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
