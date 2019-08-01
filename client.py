import cv2
from selenium import webdriver
import numpy as np
import time
import zmq

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

# video read and display
while(True):
    ret1, frame1 = cap1.read()
    frame1 = cv2.resize(frame1, (374, 256))

    socket.send_pyobj(frame1)
    #socket.send(frame1.tobytes())
    print("send", end=' ')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    
    # wait for return
    frame_result = socket.recv_pyobj()
    #frame_result = np.frombuffer(socket.recv(), dtype=np.uint8).reshape(256, 374, 3)
    #cv2.namedWindow('from server', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("from server", 480, 270)
    #cv2.moveWindow("from server", size["width"],0)

    print(frame_result)
    a, b = frame_result
    y, x = round(-50*a), round(-50*b)
    cv2.arrowedLine(frame1, (187, 128), (x+187, y+128), ( 0, 255, 0))
    cv2.imshow('from server', frame1)

    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release camera
cap1.release()
# close web page
browser.quit()

# close OpenCV windows
cv2.destroyAllWindows()