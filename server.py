import zmq
import cv2
import io
import numpy as np
import dlib
import time
import imutils
from PIL import Image
from test import run


# HOST = '140.114.77.221'
# PORT = 3521
ADDRESS = 'tcp://*:3521'

# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# s.bind((HOST, PORT))
# s.listen(5)
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind(ADDRESS)

# default face detector
detector = dlib.get_frontal_face_detector()
# load in model
predictor = dlib.shape_predictor( 'shape_predictor_68_face_landmarks.dat')

while True:
    # conn, addr = s.accept()
    # print ('Connected by %s:%s' % (addr[0], addr[1]))
    print("Started...........")
    while True:
        frame1 = socket.recv_pyobj()
        #frame1 = np.frombuffer(socket.recv(), dtype=np.uint8).reshape(256, 374, 3)
        print("received", end=' ')
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        
        # detect face
        face_rects, scores, idx = detector.run(frame1, 0)

        print(scores)
        if len(face_rects) == 0:
            socket.send_pyobj([0, 0])
            continue
        
        face = face_rects[np.array(scores).argmax()]

        #for i, d in enumerate(face_rects):
        #print(d)
        # color of certain frame
        # landmarks_frame = cv2.cvtColor(frame1, cv2. COLOR_BGR2RGB)

        # # find the feature point's position
        # shape = predictor(landmarks_frame, d)
    
        # # draw 68 feature point
        # for i in range(36, 48):
        #     cv2.circle(frame1,(shape.part(i).x,shape.part(i).y), 3,( 0, 0, 255), 2)
        #     cv2.putText(frame1, str(i),(shape.part(i).x,shape.part(i).y),cv2. FONT_HERSHEY_COMPLEX, 0.5,( 255, 0, 0), 1)

        #bounding box's corner position
        # x1 = max(shape.part(i).x for i in range(36, 48))
        # y1 = max(shape.part(i).y for i in range(36, 48))
        # x2 = min(shape.part(i).x for i in range(36, 48))
        # y2 = min(shape.part(i).y for i in range(36, 48))
        #i, d = face_rects[face]
        H, W, _ = frame1.shape
        x1 = max(face.left(), 0)
        y1 = max(face.top(), 0)
        x2 = min(face.right(), W)
        y2 = min(face.bottom(), H)

        # bounding box
        #cv2.rectangle(frame1, (x1+5, y1+5), (x2-5, y2-5), ( 0, 255, 0), 2, cv2. LINE_AA)
        img = frame1[y1:y2, x1:x2]
        cv2.rectangle(frame1, (x1, y1), (x2, y2), ( 0, 255, 0), 4, cv2. LINE_AA)

        print(x1, x2, y1, y2)
        # cv2.imwrite('frame.jpg', frame1)
        # cv2.imwrite('crop.jpg', img)
        #cv2.resize(img, (256, 256))
        #print(frame1)

        angle = run(cv2.resize(img, (256, 256)))
        angle = angle.numpy().tolist()    
        socket.send_pyobj(angle)
        #socket.send_pyobj(frame1)
        #socket.send(frame1.tobytes())