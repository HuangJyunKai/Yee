# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 13:33:58 2019

@author: 07067
"""

import numpy as np
import cv2
import json

def fun_box2hw(bbox_axis):
    '''
     The bounding box is a rectangular box 
     that can be determined by the  x  and  y  axis coordinates 
     in the upper-left corner and the  x  and  y  axis coordinates 
     in the lower-right corner of the rectangle.
    '''
    # bbox: bounding box (left top and right bottom)
    bbox_w = bbox_axis[2]-bbox_axis[0]
    bbox_h = bbox_axis[3]-bbox_axis[1]
    x_bbox_center = int((bbox_axis[0]+bbox_axis[2])/2)
    y_bbox_center = int((bbox_axis[1]+bbox_axis[3])/2)
#    bbox_center = [int((bbox_axis[0]+bbox_axis[2])/2),
#                   int((bbox_axis[1]+bbox_axis[3])/2)]
    return [x_bbox_center, y_bbox_center, bbox_w, bbox_h]

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def fun_imgOpening(img):
    '''
    Opening is just another name of erosion followed by dilation
    It is useful in removing noise, as we explained above. Here we use the function
    '''
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return opening


def fun_MaskPointProb2Points(mask_point):
    '''
    make_point: H*W*3(RGB)
    return: corner points
    '''
    # RGB2Gray
    mask_point_p = rgb2gray(mask_point)
    mask_point_p = np.uint8(mask_point_p)
    # Binarization
    th, threshed = cv2.threshold(mask_point_p, 1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # opening
    threshed = fun_imgOpening(threshed)
    # findcontours
    cnts = cv2.findContours(threshed, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2]
    # filter by area 1~100
    s1 = 1
    s2 = 100
    xcnts = []
    for cnt in cnts:
        if s1 < cv2.contourArea(cnt) < s2:
            #find center
            maxcenter_x = max(cnt[:,:,0])
            mincenter_x = min(cnt[:,:, 0])
            center_x = int((maxcenter_x+mincenter_x)/2)
            maxcenter_y = max(cnt[:, :, 1])
            mincenter_y = min(cnt[:, :, 1])
            center_y =int((maxcenter_y + mincenter_y) / 2)
            xcnts.append([center_x, center_y])
    return xcnts

def fun_detrend(x,**kwargs):
        '''
        %DETREND Remove a linear trend from a vector, usually for FFT processing.
        %   Y = DETREND(X) removes the best straight-line fit linear trend from 
        %   the data in vector X and returns it in vector Y.  If X is a matrix,
        %   DETREND removes the trend from each column of the matrix.
        %
        %   Y = DETREND(X,'constant') removes just the mean value from the vector X,
        %   or the mean value from each column, if X is a matrix.
        %
        %   Y = DETREND(X,'linear',BP) removes a continuous, piecewise linear trend.
        %   Breakpoint indices for the linear trend are contained in the vector BP.
        %   The default is no breakpoints, such that one single straight line is
        %   removed from each column of X.
        '''
        for name, value in kwargs.items():
            if name == 'o': o=value
            if name == 'bp': bp=value      
        try:    o
        except: o = 1
        try:    bp
        except: bp=0
        
        ndim=x.ndim
        if ndim==1: # If a row, turn into column vector
            N=np.shape(x)[0]
        elif ndim==2:
            nvar, nobs=np.shape(x)
            if (nvar<nobs):
                x=np.transpose(x)
            N, nvar=np.shape(x)
        
        if (o is 'constant') or (o is 'c') or (o is 0): #Remove just mean from each column
            y= x - np.kron(np.ones((N,1)),np.mean(x,axis=0))
        elif (o is 'linear') or (o is 'l') or (o is 1):
            bp = np.unique([0,bp,N-1]) # Include both endpoints
            lb = len(bp)-1
            a = np.concatenate((np.zeros((N,lb),dtype=x.dtype),np.ones((N,1),dtype=x.dtype)),axis=1)
            for kb in range(lb):
                M = N - bp[kb]
                a[range(M)+bp[kb],kb]=(np.array(range(M))+1)/M;
            b=np.linalg.lstsq(a,x)[0]
            y=x-np.dot(a,b) # Remove best fit
            # Build regressor with linear pieces + DC    
        if ndim == 1:
          y = y.transpose()
        return y
    
def fun_json2ObjLoc(json_filepath):
    with open(json_filepath, 'r') as f:
        data = json.load(f) 
    bboxes=[]
    for shape in data['shapes']:
        if shape['shape_type']=='rectangle':
            label = shape['label']
            points = shape['points']
            box={}
            box['label']=label
            x_min = min(points[0][0], points[1][0])
            y_min = min(points[0][1], points[1][1])
            x_max = max(points[0][0], points[1][0])
            y_max = max(points[0][1], points[1][1])
#            box['points']=[points[0][0],points[0][1], points[1][0], points[1][1]]
            box['points']=[int(x_min),int(y_min),int(x_max), int(y_max)]
            bboxes.append(box)
    return bboxes