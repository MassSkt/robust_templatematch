import cv2
import numpy as np
import gc
import itertools
from joblib import Parallel, delayed
import os

## contour calculation ####
def calc_contours(img,remove_edge=True):
  #img=cv2.erode(img,np.ones((erode_kernel,erode_kernel)),iterations=1)
  (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

  if int(major_ver) < 4 :
      '''
          Old OpenCV 2 code goes here
      '''
      img,contours,hierarchy=cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # choose external to extract remove internal contour
  else :
      '''
          New OpenCV 3 code goes here 
      '''
      contours,hierarchy=cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  
  ret_contours=[]
  for cnt in contours:
      # omit rects on the edge of the image
      if remove_edge==True:
          if (not (np.any(cnt==0) or np.any(cnt[:,:,1]==img.shape[0]-1) or np.any(cnt[:,:,0]==img.shape[1]-1))):
              ret_contours.append(cnt)
      else:
          ret_contours.append(cnt)

  return ret_contours


def draw_contours(img,contours,min_area,min_len):
    ret_img=img.copy()
    ret_contours=[]
    for cnt in contours:
        if len(cnt) > 0:

            # remove small objects
            if cv2.arcLength(cnt, True) < min_len:
                continue
            if cv2.contourArea(cnt) < min_area:
                continue
            ret_contours.append(cnt)
            #cv2.polylines(img, np.int32(contours[i].transpose(1, 0, 2)), True, (255, 0, 0), 5)
    cv2.drawContours(ret_img, ret_contours, -1, (255, 0, 0), 2)
            #ret_img = cv2.polylines(img, contours[i].transpose(1, 0, 2), isClosed=True, color=(0, 255, 0), thickness=5)
    return ret_img

def draw_boxes(img,list_of_loc_size_tuple):
    ret_img=img.copy()
    for w_idx,h_idx,w,h in list_of_loc_size_tuple:
         ret_img=cv2.rectangle(ret_img,(w_idx,h_idx),(w_idx+w,h_idx+h),(0,255,0),2)
    return ret_img
    
def approximate_contours(contours,min_area,min_len):
    '''
    approximate contours to rough contour
    '''
    
    approx_contours = []
    for i, cnt in enumerate(contours):
        if len(contours[i]) > 0:

            # remove small objects
            if cv2.arcLength(contours[i], True) < min_len:
                continue
            if cv2.contourArea(contours[i]) < min_area:
                continue
        
        approx_contours.append(cnt)
        # 元の輪郭及び近似した輪郭の点の数を表示する。
        #print('contour {}: {} -> {}'.format(i, len(cnt), len(approx_cnt)))

    return approx_contours

def get_min_rect_boxes_contour(contours,return_loc_and_angle_info=False):
    ret_contours=[]
    ret_loc_and_angle=[]
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        ret_loc_and_angle.append(rect) # rect ((w_idx,h_idx),(w,h),angle)
        box = cv2.boxPoints(rect)[:,None,:]#reshape to contour format
        box = np.int0(box)
        #im = cv2.drawContours(im,[box],0,(0,0,255),2)
        ret_contours.append(box)
    if return_loc_and_angle_info==False:
        return ret_contours
    else:
        return ret_contours,ret_loc_and_angle


def get_rect_boxes_location(contours):
    ret_location=[]
    for cnt in contours:
        w_idx,h_idx,w,h = cv2.boundingRect(cnt)
        ret_location.append((w_idx,h_idx,w,h))
    return ret_location

