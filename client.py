import cv2
import time
import zmq
import dlib
import numpy as np

from selenium import webdriver
from PIL import ImageGrab
from pos_cal import cal_pos

# HOST = '140.114.77.221'
# PORT = 3521
ADDRESS = 'tcp://140.114.77.221:3521'

# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# s.connect((HOST, PORT))
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect(ADDRESS)

cap_num = 0
cap1 = cv2.VideoCapture(cap_num)

# website address
#browser=webdriver.Safari()
#browser.get("http://ocw.nthu.edu.tw/ocw/index.php?page=chapter&cid=242&chid=2649")
# browser.get("https://www.google.com")
# browser.set_window_position(0, 0)
# browser.set_window_size(512, 380)
# size = browser.get_window_size()

if not cap1.isOpened():
    cap1.open(cap_num)

# default face detector
detector = dlib.get_frontal_face_detector()
# load in model
predictor = dlib.shape_predictor( 'shape_predictor_68_face_landmarks.dat')

tune = []

# calibration, video read and display 
for num in range(5):
    while(True):
        ret1, frame1 = cap1.read()
        #frame1 = cv2.resize(frame1, (374, 256))
        
        # detect face
        face_rects, scores, idx = detector.run(frame1, 0)
        
        if len(face_rects) == 0:
            continue
        
        face = face_rects[np.array(scores).argmax()]

        H, W, _ = frame1.shape
        x1 = max(face.left(), 0)
        y1 = max(face.top(), 0)
        x2 = min(face.right(), W)
        y2 = min(face.bottom(), H)

        # bounding box
        img = frame1[y1:y2, x1:x2]
        cv2.rectangle(frame1, (x1, y1), (x2, y2), ( 0, 255, 0), 4, cv2. LINE_AA)
        #print(x1, x2, y1, y2)

        img = cv2.resize(img, (256, 256))
        socket.send_pyobj(img)
        #socket.send(frame1.tobytes())
        print("send", end=' ')
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        
        # wait for return
        frame_result = socket.recv_pyobj()
        #frame_result = np.frombuffer(socket.recv(), dtype=np.uint8).reshape(256, 374, 3)
        cv2.namedWindow('from server', cv2.WINDOW_NORMAL)
        cv2.moveWindow("from server", 0,0)
        #cv2.setWindowProperty('from server', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        #cv2.resizeWindow("from server", W, H)
        #x_pos, y_pos, w, h =cv2.getWindowImageRect('from server')

        #print(frame_result)
        a, b = frame_result
        #y, x = round(-500*a), round(-500*b)
        
        #cv2.arrowedLine(frame1, (W//2, H//2), (W//2+x, H//2+y), (255, 0, 0), 5, cv2.LINE_AA, 0, 0.1)
        
        # mode calibration or detection
        if(num is 0):
            x_pos, y_pos = W//4, H//4
        elif(num is 1):
            x_pos, y_pos = W//4*3, H//4
        elif(num is 2):
            x_pos, y_pos = W//4*3, H//4*3
        elif(num is 3):
            x_pos, y_pos = W//4, H//4*3
        elif(num is 4):
            y, x = cal_pos(a, b, W, H, tune)
            x_pos, y_pos = round(W//2+x), round(y)
        
        print(x_pos, y_pos)
        
        cv2.circle(frame1, (x_pos, y_pos), 10, (255, 0, 0), -1)
        cv2.imshow('from server', frame1)

        # press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            tune.append([a, b])
            break

# release camera
cap1.release()
# close web page
browser.quit()

# close OpenCV windows
cv2.destroyAllWindows()