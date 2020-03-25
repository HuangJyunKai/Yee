import os
import torch
import numpy as np
import cv2
import time
from argparse import ArgumentParser
from PIL import Image
import src.models.Models as Net
from models.multi_tasks import ELANetV3_modified_sigapore
from src.fun_makedecision import fun_detection_TrafficViolation
from src.model_inference_ObjectDetect_Elanet import detect
from src.model_inference_Segmentation import evaluateModel
from src.fun_plotfunction import  plot_ROI, plot_bbox_Violation, plot_bbox

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load segmentation model
filepath_model_seg_LaneLine = './models/model_300_lane_512_256_p2_q3.pth'
filepath_model_seg_road = './models/model_213_road_512_256_p2_q3.pth'
# Load Object Detection model
checkpoint_path = './models/od_NoPretrain/BEST_checkpoint.pth.tar'


#pretrained_dict = {k[7:]: v for k, v in pretrain_dict.items() if k[7:] in model_dict}
#pretrained_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}



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
    pretrained_dict = {k: v for k, v in model_refernce.items() if k in model_seg_dict}
#    print(pretrained_dict.keys())
#    print(model_seg_dict.keys())
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


        
        
model_seg_lane = fun_load_Seg_model(filepath_model_seg_LaneLine, flag_seg_type='LaneLine')
model_seg_road = fun_load_Seg_model(filepath_model_seg_road, flag_seg_type='Road')
model_od = fun_load_od_model(checkpoint_path)

if __name__ == '__main__':
    img_path = 'D:\\專案管理\\新加坡專案\\label\\OV_001-1-Segmentation\\OV_001-1 0036.jpg'

    original_image = Image.open(img_path, mode='r')
    original_image = original_image.convert('RGB')

    
    # object detection model
    annotated_image_od, bboxes = detect(model_od, original_image, min_score=0.3, max_overlap=0.5, top_k=100)
    # road segmentation model
    argmax_feats_road, color_map_display_road = evaluateModel(model_seg_road, original_image, inWidth=512, inHeight=256, flag_road=1)
    # lane segmentation model
    argmax_feats_lane, color_map_display_lane = evaluateModel(model_seg_lane, original_image, inWidth=512, inHeight=256, flag_road=0)

    argmax_feats_road[argmax_feats_road==11]=100
    argmax_feats_lane[argmax_feats_lane==11]=100
    original_image = np.array(original_image)



    decision_boxes, img_result = fun_detection_TrafficViolation(original_image, bboxes, argmax_feats_lane,argmax_feats_road)
    map_seg_label_line=argmax_feats_lane 
    map_seg_label_road=argmax_feats_road
    
#    annotated_image_od_ = cv2.cvtColor(np.asarray(annotated_image_od),cv2.COLOR_RGB2BGR)
#    imfusion = cv2.addWeighted(color_map_display_road, 0.1, annotated_image_od_, 1, 0)
#    imfusion = cv2.addWeighted(color_map_display_lane, 0.5, annotated_image_od_, 1, 0)

    

    img_fusion = original_image.copy()
    
    for bbox in bboxes:
        img_fusion = plot_bbox(img_fusion, bbox)    
    img_fusion = cv2.addWeighted(img_fusion, 1, color_map_display_road, 0.5, 0)   
    img_fusion = cv2.addWeighted(img_fusion, 1, color_map_display_lane, 1, 0)
    
    imgs = np.hstack([img_fusion,img_result])
    imgs = imgs[...,[2,1,0]]
    cv2.imshow('',cv2.resize(imgs,(1440,405)))