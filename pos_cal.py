import math
import numpy as np

def cal_pos(pitch, yaw, w, h, tune):
    pos_to_pixel = 30
    
    x = (tune[1][1]-yaw)/(tune[1][1]-tune[0][1])*(w/4)+(yaw-tune[0][1])/(tune[1][1]-tune[0][1])*(w/4*3)
    y = (tune[2][0]-pitch)/(tune[2][0]-tune[1][0])*(h/4)+(pitch-tune[1][0])/(tune[2][0]-tune[1][0])*(h/4*3)
    # y = -math.tan(pitch)*50*pos_to_pixel
    # x = -math.tan(yaw)*50*pos_to_pixel

    return y, x
    