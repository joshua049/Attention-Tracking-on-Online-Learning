import cv2
from selenium import webdriver
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
browser=webdriver.Safari()
browser.get("http://ocw.nthu.edu.tw/ocw/index.php?page=chapter&cid=242&chid=2649")
browser.set_window_position(0, 0)
browser.set_window_size(512, 380)
size = browser.get_window_size()

if not cap1.isOpened():
    cap1.open(cap_num)

# video read and display
while(True):
    ret1, frame1 = cap1.read()

    socket.send_pyobj(frame1)
    # cv2.imshow('client', frame1)

    frame_result = socket.recv_pyobj()
    cv2.namedWindow('from server', cv2.WINDOW_NORMAL)
    cv2.resizeWindow("from server", 480, 270)
    cv2.moveWindow("from server", size["width"],0)
    cv2.imshow('from server', frame_result)

    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release camera
cap1.release()
# close web page
browser.quit()

# close OpenCV windows
cv2.destroyAllWindows()