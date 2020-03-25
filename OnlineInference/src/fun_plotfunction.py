# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:59:44 2019

@author: 07067
"""
import numpy as np
import cv2
from src.fun_linefit_ransac import polyval


def plot_parkingline(img, linears_formula, formula_line_parkline_main, color):
    x=np.array(range((1920)))
    x=x[500:1500]
    line_2_x = np.array([500,1500])
    for linear_formula in linears_formula:     
        line_2_y=linear_formula['coefficient'][1]*line_2_x + linear_formula['coefficient'][0]
        cv2.line(img, (int(line_2_x[0]), int(line_2_y[0])), (int(line_2_x[1]), int(line_2_y[1])), color, 10 )
    if len(formula_line_parkline_main)>0:
        line_2_y=formula_line_parkline_main['coefficient'][1]*line_2_x + formula_line_parkline_main['coefficient'][0]
        cv2.line(img, (int(line_2_x[0]), int(line_2_y[0])), (int(line_2_x[1]), int(line_2_y[1])), (0, 0, 255), 10)
    return img

def plot_line(img, linears_formula,  color):
    x = linears_formula['x_range']
    if linears_formula['order']>1:
        iterval=int((x[1]-x[0])/4)
        x = [x[0], x[0]+iterval, x[0]+2*iterval,x[0]+3*iterval, x[1]]
    
    line_2_x = np.array(x)
    line_2_y = polyval(linears_formula['coefficient'],line_2_x,linears_formula['order']) 

    if linears_formula['order']>1:
        for i in range(4):
            cv2.line(img, (int(line_2_y[i]), int(line_2_x[i])), (int(line_2_y[i+1]), int(line_2_x[i+1])), (0, 0, 255), 10)
    else:
        cv2.line(img, (int(line_2_y[0]), int(line_2_x[0])), (int(line_2_y[1]), int(line_2_x[1])), (0, 0, 255), 10)
    return img

def plot_bbox(img, bbox):
    label = bbox['label']
    x_min, y_min, x_max, y_max = bbox['points']
    
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max ), (0,255,0), 3)
    cv2.putText(img, label, (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 5)
    return img

def plot_bbox_Violation(img, bbox, color):
    label = bbox['label']
    x_min, y_min, x_max, y_max = bbox['points']
    
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max ), color, 3)
    cv2.putText(img, label, (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 5)
    return img

def plot_ROI(img, roi):
    x_min, y_min, x_max, y_max = roi
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max ), (255,0,255), 3)
    return img