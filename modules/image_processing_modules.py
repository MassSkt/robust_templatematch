import cv2
import numpy as np
import gc
import itertools
from joblib import Parallel, delayed
import os

def get_edge_binary_image(img):
    # convert to gray ir rgb 
    if img.ndim==3:
        img=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_thresh= cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                    cv2.THRESH_BINARY,11,2)
    img_thresh=((img_thresh==0)*255).astype(np.uint8)

    return img_thresh

def rot_cut(src_img, deg, center, size):
    rot_mat = cv2.getRotationMatrix2D(center, deg, 1.0)
    rot_mat[0][2] += -center[0]+size[0]/2 #
    rot_mat[1][2] += -center[1]+size[1]/2 # 
    return cv2.warpAffine(src_img, rot_mat, size)

def min_rect_cut(img,cnt):
    # min rect
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    minr_cnt = np.int0(box)

    # center
    (x,y),radius = cv2.minEnclosingCircle(minr_cnt)

    # cut with rotation
    cut_img=rot_cut(img.copy(),rect[2],(x,y),(int(rect[1][0]),int(rect[1][1])))

    return cut_img