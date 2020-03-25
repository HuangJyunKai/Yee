# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 11:24:32 2019

@author: 07067
"""
import json
import cv2
import numpy as np
from PIL import Image,ImageDraw

'''
Line                        RGB
pm:機車格       #abe260: (67,89,38)
pc: 汽車格      #88cf26: (53,81,15) 
Pb: 公車格      #6b31cb: (42,19,80)
pw: 機車等待區  #8858d7: (53,35,84) 
pl: 機車待轉區  #8858d7: (53,35,84) 
zc: 斑馬線      #fffdf9: (100,99,98)
cl: 網黃線      #ffa500: (100,65,0)
rs: 紅線        #e32636: (89,15,21)
ys: 黃線 實線   #ffff00: (100,100,0)
yd: 黃線 虛線   #ffff00: (100,100,0)
ds: 雙黃線 實線 #cccc00: (80,80,0)
dd: 雙黃線 虛線 #cccc00: (80,80,0)
ws: 白線 實線   #ffffff: (100,100,100)
wd: 白線 虛線   #fef5e6: (100,96,90)

pp: 停車個角點

Seg
sw: 人行道	          #e32636:(89,15,21)
dr: 可行駛道路區域	    #ffff00:(100,100,0)
ud: 不可行駛道路區域	 #a89cfb:(66,61,98)

object detection
b: 大車(巴士/公車類/砂石車/聯結車)
t: 貨車
m: 機車 (無騎士)
k: 行駛機車(含有騎士)
c: 車

'''

line_labels_color={'pm':(67,89,38),
                   'pc':(53,81,15),
                   'Pb':(42,19,80),
                   'pw':(53,35,84),
                   'pl':(53,35,84),
                   'zc':(100,99,98),
                   'cl':(100,65,0),
                  'rs':(89,15,21),
                  'ys':(100,100,0),
                  'yd':(100,100,0),
                  'ds':(80,80,0),
                  'dd':(80,80,0),
                  'ws':(100,100,100),
                  'wd':(100,96,90)}
labels_line={'pm':0,
             'pc':1,
             'Pb':2,
             'pw':3,'pl':3,
             'zc':4,
             'cl':5,
             'rs':6,
             'ys':7,'yd':7,
             'ds':8,'dd':8,
             'ws':9,
             'wd':10}

seg_labels_color={'sw':(89,15,21),
                  'dr':(100,100,0),
                  'ud':(91,90,100)}
labels_seg={'sw':1,
            'dr':2,
            'ud':0}


point_label=['pp']
object_label=['b','t','m','k','c']



def fun_get_labels():
    confg={}
    line, seg = {},{}
    line['labels_color'] = line_labels_color
    line['labels'] = labels_line
    seg['labels_color'] = seg_labels_color
    seg['labels'] = labels_seg
    confg['line']=line
    confg['seg']=seg
    confg['seg']=seg
    confg['point']=point_label
    confg['object_label']=point_label
    
    return confg
    

    
def fun_maskimage2labelimage(mask_img, labels_color, labels):
    h,w,_=mask_img.shape
    label_mask = np.ones([h,w], dtype=np.uint8)*100
    for tmp_indx in labels_color:
        label = labels[tmp_indx]
        color_label = labels_color[tmp_indx]
        label_mask[(mask_img[:,:,0]==color_label[2]) & (mask_img[:,:,1]==color_label[1]) &(mask_img[:,:,2]==color_label[0])]=label
    return label_mask
    


def fun_GaussianDistribution(x,y,sigma):
    return (1/(2*np.pi*(sigma**2)))*np.exp(-(x**2+y**2)/(2*sigma**2))
def fun_GaussianCircle(sigma,size_x):
    size_y = size_x
    ciricle_GaussianDistribution = np.zeros((size_x*2+1,size_y*2+1))
    for i_x, x in enumerate(range(-size_x,size_x+1)):
        for  i_y, y in enumerate(range(-size_x,size_x+1)):
            ciricle_GaussianDistribution[i_x,i_y]=fun_GaussianDistribution(x,y,sigma)
    ciricle_GaussianDistribution = ciricle_GaussianDistribution/np.sum(ciricle_GaussianDistribution)
    return ciricle_GaussianDistribution

def fun_json2LinePoint(json_filepath):
    with open(json_filepath, 'r') as f:
        data = json.load(f) 
    w=data['imageWidth']
    h=data['imageHeight']  
    center_car = np.array([w/2/w, h/h])
    
    mask_line = np.zeros([h,w,3], dtype=np.uint8)
    mask_line = Image.fromarray(mask_line)
    mask_point = np.zeros([h,w,3], dtype=np.uint8)
    
    for shape in data['shapes']:
        label = shape['label']
        polygons = shape['points']
        if label in labels_line.keys():
            label_color = line_labels_color[label]
            xy = list(map(tuple, polygons))
            ImageDraw.Draw(mask_line).polygon(xy=xy, fill=label_color)
        elif label in point_label:
            point = shape['points'][0]
            point[0]=int(point[0])
            point[1]=int(point[1])
            
            point_axis=np.array([point[0]/w, point[1]/h])
            dist = np.sqrt(np.sum(center_car-point_axis)**2)

            weight = int(1 / dist)*5
            if weight > 5:
                weight = 5
            elif weight==0:
                weight=1
                
            cir_Gaussian = fun_GaussianCircle(weight/2,weight)
            cir_Gaussian = np.uint8(cir_Gaussian/np.max(cir_Gaussian)*255)
            for i_x, x in enumerate(range(point[0]-weight,point[0]+weight+1)):
                for i_y, y in enumerate(range(point[1]-weight,point[1]+weight+1)): 
                    if (x>=0) & (x< h):
                       if (y>=0) & (y< w): 
                           mask_point[y,x,:] = cir_Gaussian[i_x,i_y]      
    

               
    mask_line = mask_line.convert('RGB')
    mask_line = np.array(mask_line)
    mask_line = cv2.cvtColor(mask_line,cv2.COLOR_RGB2BGR) 
    return mask_line, mask_point

def fun_json2SegImage(json_filepath, flag_background=1):
    with open(json_filepath, 'r') as f:
        data = json.load(f) 
    w=data['imageWidth']
    h=data['imageHeight']  
    
    if flag_background==1:
        mask_seg = np.ones([h,w,3], dtype=np.uint8)
        color_background = seg_labels_color['ud'] # disable driving area
        mask_seg[:,:,0]*=color_background[2]
        mask_seg[:,:,1]*=color_background[1]
        mask_seg[:,:,2]*=color_background[0]
    else:
        mask_seg = np.zeros([h,w,3], dtype=np.uint8)
    mask_seg = Image.fromarray(mask_seg)
    
    for shape in data['shapes']:
        label = shape['label']
        polygons = shape['points']
        if label in labels_seg.keys():
            label_color = seg_labels_color[label]
            xy = list(map(tuple, polygons))
            ImageDraw.Draw(mask_seg).polygon(xy=xy, fill=label_color)
    mask_seg = mask_seg.convert('RGB')
    mask_seg = np.array(mask_seg)
    mask_seg = cv2.cvtColor(mask_seg,cv2.COLOR_RGB2BGR) 
    return mask_seg

def fun_json2Object(json_filepath):
    with open(json_filepath, 'r') as f:
        data = json.load(f) 
    w=data['imageWidth']
    h=data['imageHeight']
    mask = np.zeros([h,w,3], dtype=np.uint8)
    mask = Image.fromarray(mask)
    objects={}
    c=1
    for shape in data['shapes']:
        label = shape['label']
        polygons = shape['points']
        objects[c]={'object':label, 'box':[int(polygons[0][0]), int(polygons[0][1]),int(polygons[1][0]), int(polygons[1][1])]}
        c+=1
    return objects
    
#if __name__ == '__main__':
#    json_filepath = 's_road.json' # s_road, s_line     
#    mask_road = fun_json2SegImage(json_filepath)
#    json_filepath = 's_line.json' # s_road, s_line     
#    mask_line = fun_json2SegImage(json_filepath)
#    mask_point=fun_json2SegPoint('s_point.json')
#      
#    img = cv2.imread("s.jpg")
#    mg_mix_road = cv2.addWeighted(img, 0.7, mask_road, 0.7, 0)
#    mg_mix_line = cv2.addWeighted(img, 0.7, mask_line, 0.7, 0)
#    mg_mix_point = cv2.addWeighted(img, 0.7, mask_point, 0.7, 0)
#    
#    objects = fun_json2Object('s_objects.json')
#img=Image.fromarray(np.uint8(img))
#for i in range(3):
#    img[:,:,i]+=(mask)
    
#cv2.imshow('',mg_mix_point)
#cv2.imshow('',mg_mix_road)

