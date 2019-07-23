import zmq
import cv2
import io
import numpy as np
import dlib
import imutils
from PIL import Image


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
        print("received")
        
        # detect face
        face_rects, scores, idx = detector.run(frame1, 0)

        for i, d in enumerate(face_rects):
            # color of certain frame
            landmarks_frame = cv2.cvtColor(frame1, cv2. COLOR_BGR2RGB)

            # find the feature point's position
            shape = predictor(landmarks_frame, d)
        
            # draw 68 feature point
            for i in range(36, 48):
                cv2.circle(frame1,(shape.part(i).x,shape.part(i).y), 3,( 0, 0, 255), 2)
                cv2.putText(frame1, str(i),(shape.part(i).x,shape.part(i).y),cv2. FONT_HERSHEY_COMPLEX, 0.5,( 255, 0, 0), 1)

            #bounding box's corner position
            x1 = max(shape.part(i).x for i in range(36, 48))
            y1 = max(shape.part(i).y for i in range(36, 48))
            x2 = min(shape.part(i).x for i in range(36, 48))
            y2 = min(shape.part(i).y for i in range(36, 48))

            # bounding box
            cv2.rectangle(frame1, (x1+2, y1+2), (x2-2, y2-2), ( 0, 255, 0), 2, cv2. LINE_AA)
        
        socket.send_pyobj(frame1)