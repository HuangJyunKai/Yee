# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 09:51:16 2019

@author: 07067
"""
import os
import torch
import numpy as np
import cv2
import time
from argparse import ArgumentParser
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
import threading 
from queue import Queue
import src.models.Models as Net
from models.multi_tasks import ELANetV3_modified_sigapore
from src.fun_makedecision import fun_detection_TrafficViolation
from src.model_inference_ObjectDetect_Elanet import detect
from src.model_inference_Segmentation import evaluateModel, evaluateModel_models
from src.fun_plotfunction import  plot_ROI, plot_bbox_Violation, plot_bbox
from src.fun_modify_seg import fun_intergate_seg_LaneandRoad
from src.violation import save_violation, createFolder
from IPython.display import clear_output

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('GPU:',device)
# load segmentation model
filepath_model_seg_LaneLine = './models/model_148_lane_512_256_p2_q3_1206.pth'
filepath_model_seg_road = './models/model_147_road_512_256_p2_q3_1206.pth'
# Load Object Detection model
checkpoint_path = './models/od_NoPretrain/BEST_checkpoint.pth.tar'
video_path='./mnt/OV_001-Part1_OV_001_0001.avi'
savefilename='OV_001-Part1_OV_001_0001'

video_t_start = 0 # unit: second
video_t_end = 60 # unit: second

#partial_inference_video =[[0,20],[20,40],[40,60]]
partial_inference_video =[[0,1*60],[1*60, 2*60],[2*60, 3*60]]


Tensor = torch.cuda.FloatTensor

createFolder('./violation')

def fun_capture_fileapth(folder_path1, folder_path2):
    filename_path1=[]
    filename_path2=[]
    for tmp in os.listdir(folder_path1):
        if os.path.splitext(tmp)[1] == ".json":
            filename_path1.append(tmp)
    for tmp in os.listdir(folder_path2):
        if os.path.splitext(tmp)[1] == ".json":
            filename_path2.append(tmp)
    filename_path_result1=[]
    filename_path_result2=[]
    for tmp1 in filename_path1:
        flag=0
        for tmp2 in filename_path2:
            if tmp1==tmp2:
                flag=1
        if flag==1:
            filename_path_result1.append(os.path.join(folder_path1, tmp1))
            filename_path_result2.append(os.path.join(folder_path2, tmp1))
    return filename_path_result1, filename_path_result2
            

def fun_load_Seg_model(filepath_model_seg, flag_seg_type):  
    if not os.path.isfile(filepath_model_seg):
        print('Pre-trained model (Lane Line Segmentation) file does not exist !!!!!')
        exit(-1)
    if flag_seg_type == 'LaneLine':
        model_seg = Net.ESPNet(classes=12, p=2, q=3) 
    elif flag_seg_type == 'Road':
        model_seg = Net.ESPNet_corner_heatmap(classes=3, p=2, q=3)  
    model_seg_dict = model_seg.state_dict()
    model_refernce = torch.load(filepath_model_seg, map_location=device)
    #pretrained_dict = {k[7:]: v for k, v in pretrain_dict.items() if k[7:] in model_dict}
    pretrained_dict = {k: v for k, v in model_refernce.items() if k in model_seg_dict}

    model_seg_dict.update(pretrained_dict)
    model_seg.load_state_dict(model_seg_dict)
    
    model_seg = model_seg.to(device)   
    model_seg.eval()
    print('load model (Lane Line Segmentation) : successful')
    return model_seg


def fun_load_od_model(checkpoint_path):
    model_od = ELANetV3_modified_sigapore.SSD352(n_classes=6)
    model_od_dict = model_od.state_dict()
    model_refernce = torch.load(checkpoint_path, map_location=device)
    model_refernce = model_refernce['model'].state_dict()
    pretrained_dict = {k: v for k, v in model_refernce.items() if k in model_od_dict}
#    print(pretrained_dict.keys())
#    print(model_od_dict.keys())
    model_od_dict.update(pretrained_dict)
    model_od.load_state_dict(model_od_dict)
    model_od = model_od.to(device)
    
    model_od.eval()
    print('load model (Object detection) : successful')
    return model_od

def thread_detect(model_od, frame_pil_img, q_detect):
    _, bboxes = detect(model_od, frame_pil_img, min_score=0.65, max_overlap=0.5, top_k=50,device=device)
    q_detect.put(bboxes)

def thread_seg_road(model_seg_road, frame_pil_img, q_sed_road):
    argmax_feats_road, color_map_display_road = evaluateModel(model_seg_road, frame_pil_img, inWidth=512, inHeight=256, flag_road=1)
    q_sed_road.put([argmax_feats_road,color_map_display_road])

def thread_seg_line(model_seg_line, frame_pil_img, q_sed_line):
    argmax_feats_road, color_map_display_road = evaluateModel(model_seg_line, frame_pil_img, inWidth=512, inHeight=256, flag_road=0)
    q_sed_line.put([argmax_feats_road,color_map_display_road])
    
def thread_seg_models(model_seg_road, model_seg_lane ,frame_pil_img, q_sed):
    argmax_feats_road,argmax_feats_lane, color_map_display_road, color_map_display_lane = evaluateModel_models(model_seg_road,model_seg_lane, frame_pil_img, inWidth=512, inHeight=256, device=device)
    q_sed.put([argmax_feats_road,argmax_feats_lane, color_map_display_road, color_map_display_lane])
    
        
model_seg_lane = fun_load_Seg_model(filepath_model_seg_LaneLine, flag_seg_type='LaneLine')
model_seg_road = fun_load_Seg_model(filepath_model_seg_road, flag_seg_type='Road')
model_od = fun_load_od_model(checkpoint_path)

q_detect = Queue()   # 宣告 Queue 物件
q_sed = Queue()   # 宣告 Queue 物件


videoCapture1 = cv2.VideoCapture(video_path)
#fps = int(videoCapture1.get(cv2.CAP_PROP_FPS))
fps = int(25)
size = (int(videoCapture1.get(cv2.CAP_PROP_FRAME_WIDTH)/1),
        int(videoCapture1.get(cv2.CAP_PROP_FRAME_HEIGHT)/1))

img_stack = []
violation_frame = -1
c=0

############################ SORT ######################################
from src.sort import Sort
from src.utils import fun_box2hw
import matplotlib.pyplot as plt

cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
mot_tracker = Sort() 
############################ SORT ######################################


for c_time, tmp_time in enumerate(partial_inference_video):
    savevideoname = savefilename + '_part{}.avi'.format(c_time)
    #DIVX (avi)
    videoWriter = cv2.VideoWriter(savevideoname,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    video_t_start = tmp_time[0]
    video_t_end = tmp_time[1]

    success, frame_np_img = videoCapture1.read()

    while success:
        if (c >= video_t_start*fps) & (c<=video_t_end*fps):
            print('{} frame:'.format(c))
            st_st=time.time()
            # BGR → RGB and numpy image to PIL image
            frame_np_img = frame_np_img[...,[2,1,0]]
            frame_pil_img = Image.fromarray(frame_np_img)
             # object detection model
            t1 = threading.Thread(target = thread_detect, args=(model_od, frame_pil_img, q_detect))
            t2 = threading.Thread(target = thread_seg_models, args=(model_seg_road, model_seg_lane ,frame_pil_img, q_sed))
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            bboxes = q_detect.get()
            argmax_feats_road, argmax_feats_lane, color_map_display_road, color_map_display_lane = q_sed.get()
            argmax_feats_road[argmax_feats_road == 11] = 100
            argmax_feats_lane[argmax_feats_lane == 11] = 100
            print('inference time:{}s'.format(time.time() - st_st))

            argmax_feats_lane, argmax_feats_road = fun_intergate_seg_LaneandRoad(argmax_feats_lane, argmax_feats_road)
            decision_boxes, img_result = fun_detection_TrafficViolation(frame_np_img, bboxes, argmax_feats_lane, argmax_feats_road)
            
            t_start_decision = time.time()
            
            # RGB → BGR
            img_result = img_result[...,[2,1,0]]

            
            ############################ SORT ######################################
            if decision_boxes is not None:
                sort_box = []
                vehicle = ['b','t','m','k','c']
                vehicle_id = [0,1,2,3,4]
        
                for box in decision_boxes :
                    a_bbox = []
                    for ii in range(0,4):
                        a_bbox.append(box['bbox']['points'][ii])
                    a_bbox.append(0.99)
                    a_bbox.append(box['bbox']['score'])
                    if box['bbox']['label'] in vehicle :
                        a_bbox.append(float(vehicle.index(box['bbox']['label'])))
                    else : 
                        assert('Vehicle TYPE ERROR')
                    sort_box.append(a_bbox)
                    
                sort_box = torch.tensor(sort_box).cuda()

                tracked_objects = mot_tracker.update(sort_box.cpu())
                
                for x1, y1, x2, y2, obj_id , cls_pred in tracked_objects:
                    cls = vehicle[int(cls_pred)]
                    img_result = cv2.UMat(img_result)
                    cv2.putText(img_result, cls + ":" + str(int(obj_id)),(int(x1+30), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (225,225,0), 3)

            ############################ SORT ######################################
            
            # save img_result
            # 偵測違規
            if violation_frame == -1 :
                for i in range(len(decision_boxes)):
                    if decision_boxes[i]['decision'] != 'pass' :
                        #建立違規資料夾
                        directory = './violation/'+str(c)
                        createFolder(directory)
                        violation_frame = c
                        break

            # 存取違規的frame前後區間
            frame_num = 20
            violation_frame,img_stack = save_violation(violation_frame,img_stack,img_result,frame_num)
        
            
            videoWriter.write(img_result)
            print('decision time : {}s'.format(time.time() - t_start_decision))
            print('total time:{}s'.format(time.time() - st_st))
            success, frame_np_img = videoCapture1.read()
            c += 1
        else:
            success = 0


    videoWriter.release()
cv2.destroyAllWindows()






