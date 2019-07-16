import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    cap.open(0)

while(True):
    ret, frame = cap.read()
    cv2.imshow('frame', frame)

    # 若按下 q 鍵則離開迴圈
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放攝影機
cap.release()

# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()
