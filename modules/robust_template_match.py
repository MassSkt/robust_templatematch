
import cv2
import numpy as np

import matplotlib.pyplot as plt

from . import contour_modules
from . import image_processing_modules


def TemplateMatch(img,template):

    # convert to gray ir rgb 
    if img.ndim==3:
        img=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if template.ndim==3:
        template=cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)

    # thresholding
    template_thresh=image_processing_modules.get_edge_binary_image(template)
    img_thresh=image_processing_modules.get_edge_binary_image(img)
    #plt.imshow(img_thresh)

    # contours for template
    area_threshold=1
    contours_tmp=contour_modules.calc_contours(template_thresh)
    contours_tmp=contour_modules.approximate_contours(contours=contours_tmp,min_area=area_threshold,min_len=1)

    # contours for imput image
    area_threshold=1
    contours=contour_modules.calc_contours(img_thresh)
    contours=contour_modules.approximate_contours(contours=contours,min_area=area_threshold,min_len=1)

    # shape matching
    match_shape_threshold=0.1
    matched_contours=[cnt for cnt in contours if cv2.matchShapes(cnt,contours_tmp[0],1,0.0)<match_shape_threshold]
    #plt.imshow(contour_modules.draw_contours(img.copy(),matched_contours,0,0))
    
    minrect_boxes_contours,box_loc_angle=contour_modules.get_min_rect_boxes_contour(matched_contours,return_loc_and_angle_info=True)
    box_locations=contour_modules.get_rect_boxes_location(matched_contours)

    return minrect_boxes_contours,box_loc_angle,box_locations



