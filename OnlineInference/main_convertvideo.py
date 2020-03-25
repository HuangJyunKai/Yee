# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 09:51:16 2019

@author: 07067
"""
import os
import torch
import numpy as np
import cv2


video_path='result_online_OV_001-2.mp4'
savefilename='result2_online_OV_001-2.mp4'


w_img=1920
h_img=1080

videoCapture1 = cv2.VideoCapture(video_path) 
fps = int(videoCapture1.get(cv2.CAP_PROP_FPS))
fps = int(30)
size = (int(videoCapture1.get(cv2.CAP_PROP_FRAME_WIDTH)/2),
        int(videoCapture1.get(cv2.CAP_PROP_FRAME_HEIGHT)/2))
videoWriter = cv2.VideoWriter(savefilename, cv2.VideoWriter_fourcc(*'DIVX'), fps, size) 
 

success, frame_np_img = videoCapture1.read() 
c=0
while success :   
    videoWriter.write(frame_np_img)
    success, frame_np_img = videoCapture1.read() 

cv2.destroyAllWindows()
videoWriter.release()





