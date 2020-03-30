# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 16:11:17 2019

@author: 07067
"""
import numpy as np
import time
from src.fun_linefit_ransac import ransac, polyfit, polyval
from scipy.signal import find_peaks
from src.utils import fun_MaskPointProb2Points, fun_box2hw
from src.fun_plotfunction import  plot_ROI, plot_bbox_Violation, plot_bbox, plot_parkingline, plot_line



def fun_fit_trafficline(image_label_line):
    '''
    image_label_line: label image with original image size
    return linear formula with polynomial fit with order = 2 
    pm: 機車格      0
    pc: 汽車格      1   Pb 公車格: 2
    pw 機車等待區   3   pl 機車待轉區 3  
    rs: 紅線        6
    ys: 黃線 實線   7   yd: 黃線 虛線   7
    ds: 雙黃線 實線 8   dd: 雙黃線 虛線 8
    ws: 白線 實線   9   wd: 白線 虛線   10
    '''
    order=2
    line_formula={}
    for tmp_label in [6,7,8,9,10]:
        pos=np.where(image_label_line==tmp_label)
        line_formula[tmp_label]={}
        if len(pos[0])>0:
            x_axis=np.array(pos[0])
            y_axis=np.array(pos[1])
            x_range=[np.min(x_axis),np.max(x_axis)]
    #        Beta_hat, y_hat_ransac,res=ransac(x_axis,y_axis,order=order,Iter=100, sigma=1, Nums=10)
#            mu = np.mean(x_axis)
#            x_axis=x_axis-mu     
            Beta_hat=polyfit(x_axis,y_axis,order)
            line_formula[tmp_label]['coefficient']=Beta_hat
            line_formula[tmp_label]['x_range']=x_range
#            line_formula[tmp_label]['mu']=mu
            line_formula[tmp_label]['order']=order
    return line_formula


def fun_fitline_bypoints(points):
    '''
    parking_points: parking_points positions
    return linear formula with polynomial fit with order = 1
    '''
    line_formula={}
    x_axis=np.array(points[:,0])
    y_axis=np.array(points[:,1])
#    mu = np.mean(x_axis)
#    x_axis=x_axis-mu     
    Beta_hat=polyfit(x_axis,y_axis,1)
    line_formula['coefficient']=Beta_hat
#    line_formula['mu']=mu
    line_formula['order']=1
    return line_formula

def fun_object_position_linear(each_line_formula, object_axis):
    # each_line_formula: dict, with keys 'coefficient','x_range','mu', 'order'
    # object_axis=[0,0]: x-axis, y-axis
    object_axis = np.array(object_axis)
    beta_hat = each_line_formula['coefficient'] 
#    mu = each_line_formula['mu'] 
    order = each_line_formula['order'] 
    x=object_axis[0]
    y=object_axis[1]
#    dist = polyval(beta_hat,x-mu,order) - y
    dist = polyval(beta_hat,x,order) - y
    if dist<0:
        return "negative", dist
    else:
        return "postive", dist
    
    
def fun_find_ParkingSpace(image_label_line,parking_points, decision_label):
    h_img,w_img = image_label_line.shape
    # pos_motor: 車道線Segmentation的結果屬於"機車停車格"或是"汽車停車格"或是"公車停車格子"
    pos_motor = np.where(image_label_line==decision_label)
    
    n_parking_point = 0
    if len(parking_points)>0: n_parking_point,_ = np.shape(parking_points)
    formulas_space_parking = {}
    formula_line_parkline_main=[]
    linears_formula=[]
    # 主停車線點數少於兩個也不計算
    # Segmentation當類別的結果太少就不計算
    if (n_parking_point<2) | (np.size(pos_motor)<100):
        # 利用角點算出主停車線
        return formulas_space_parking, linears_formula, formula_line_parkline_main
    else:
        formula_line_parkline_main = fun_fitline_bypoints(parking_points)
  
        # 假設停車格的線是橫向的，所以對"高"算直方圖
        # 然後從直方圖切區段出來，算"橫向線"的公式
        y = pos_motor[0]
        x = pos_motor[1]
        tmp_img = np.zeros((h_img,w_img))
        tmp_img[:,:] = 0
        tmp_img[y,x] = 1
        hist = np.sum(tmp_img,axis=1,dtype=np.float)
        peaks,_ = find_peaks(hist,10,distance=10)
        
        peaks_select=[]
        for point in parking_points:
            peaks_select.append(peaks[np.argmin(np.abs(point[1] - peaks))])
        peaks_select = np.array(peaks_select)
        peaks   = -np.sort(-peaks_select) #由高到低排序 
        
        for peak in peaks:
            tolerance_range = np.round(h_img/10)
            range_up = int(peak/tolerance_range)
            range_low = int(peak/tolerance_range)
#            diff_trend = np.diff(np.diff(hist[peak-30:peak+30]))
#            peaks_diff_trend,_ = find_peaks(diff_trend,5,distance=3)
#            if len(peaks_diff_trend)==2:
#                range_up = peaks_diff_trend[1]
#                range_low = peaks_diff_trend[0]        
#            else:
#                diff_trend = np.diff(hist[peak-30:peak+30])
#                for ind in range(30):
#                    if diff_trend[30+ind]>=0:
#                        range_up=ind
#                        break
#                for ind in range(30):
#                    if diff_trend[29-ind]<=0:
#                        range_low=ind
#                        break  
                
            #停車格角點的y座標有沒有若在y_min和y_max
            y_min = peak-range_low # tolerance: 10 pixels
            y_max = peak+range_up+3 # tolerance: 10 pixels
            pos_parking_point=np.where((parking_points[:,1]>=y_min) & (parking_points[:,1]<=y_max))[0]
            pos_point=[]
            if len(pos_parking_point)>0:
                parking_point = parking_points[pos_parking_point[0],:]
                reference_x=parking_point[0]
                reference_y=parking_point[1]
                pos_point=np.where((y>=y_min) & (y<=y_max) & (np.abs(reference_y-y)<30))[0]
            
            # 橫向線的點要多於500個才算有效線條 
            if len(pos_point)>10: 
                x_axis=x[pos_point]
                y_axis=y[pos_point]
                Beta_hat=polyfit(x_axis,y_axis,1)
                if np.abs(Beta_hat[1])<0.1: # 橫向線的斜率太大，可能誤判所以不採用
                    linear_formula={}
                    linear_formula['coefficient']=Beta_hat
                    linear_formula['order']=1
                    linears_formula.append(linear_formula)
            if len(pos_point)!=0:
                x=np.concatenate((x[:pos_point[0]], x[pos_point[-1]+1:]))
                y=np.concatenate((y[:pos_point[0]], y[pos_point[-1]+1:]))
    
        for i_peak_point in range(n_parking_point):
            low_bound = 0
            up_bound  = h_img
            if i_peak_point>0:
                up_bound  = parking_points[i_peak_point-1]
                low_bound = parking_points[i_peak_point]
                space_parking = (up_bound+low_bound)/2
                position_past=''
                for i_line, each_line_formula in enumerate(linears_formula):
                    position_new, _=fun_object_position_linear(each_line_formula, space_parking)
                    if ((position_past=='postive') & (position_new=='negative')) | ((position_past=='negative') & (position_new=='postive')):
                       formulas_space_parking[i_peak_point-1]={}
                       formulas_space_parking[i_peak_point-1]['formula_line_parkline_main']=formula_line_parkline_main
                       formulas_space_parking[i_peak_point-1]['formula_line_parkline_side_low']=linears_formula[i_line]
                       formulas_space_parking[i_peak_point-1]['formula_line_parkline_side_upper']=linears_formula[i_line-1]
                    position_past = position_new
        return formulas_space_parking, linears_formula, formula_line_parkline_main
    

def fun_getimageWH(image):
    if  len(image.shape)==3:
        h_img, w_img,_ = image.shape    
    else:
        h_img, w_img = image.shape
    return h_img, w_img
    
    
def fun_result_laneline_ROI_modify(image_label_line):
    # image_label_line,
    # 2.1 ROI
    h_img, w_img = fun_getimageWH(image_label_line)
    half_h_img=int(h_img/4)
    half_w_img=int(w_img/4)
    image_label_line[0:half_h_img,:]=100 # 100 background
    image_label_line[:,0:half_w_img]=100 # 100 background
    return image_label_line

def fun_result_point_ROI_modify(mask_point):
    # mask_point
    # 2.1 ROI
    h_img, w_img = fun_getimageWH(mask_point)
    half_h_img=int(h_img/4)
    half_w_img=int(w_img/4)
    mask_point[0:half_h_img:,:]=0 # 0 background
    mask_point[::,0:half_w_img]=0 # 0 background
    return mask_point


def fun_carLaneLine(image_label_line, mask_point):
    #此function用來推論"線的方程式"，所以需要"線的segmentation"和"角點的segmentation"
    h_img, w_img = fun_getimageWH(image_label_line)
    # image_label_line, mask_point
    # 2.1 ROI mask預測結果圖
    image_label_line = fun_result_laneline_ROI_modify(image_label_line)
    mask_point = fun_result_point_ROI_modify(mask_point)

    # 2.2  labeled image (line segmentation) to line function 
    # for fit the traffic lines with line predicted segmentation image
    line_traffic_formula = fun_fit_trafficline(image_label_line)
    # 2.3 corner point segmentation image to find specific points
    parking_points = fun_MaskPointProb2Points(mask_point)
    parking_points = np.array(parking_points)
    n_parking_point = 0
    if len(parking_points)>0: n_parking_point,_ = np.shape(parking_points)
    
    # 推論停車格線和紅線的線性方程式 
    # color_image用來呈現推論出"線方程式的結果用"
    color_image=np.zeros((h_img,w_img,3),dtype=np.uint8)
    if np.sum(image_label_line==0)>0: 
        # 機車停車格       
        parkingSpace_motor, linears_formula_motor, formula_line_parkline_main_motor = fun_find_ParkingSpace(image_label_line,parking_points, 0)
        color_image = plot_parkingline(color_image, linears_formula_motor, formula_line_parkline_main_motor, (255,0,255))
    elif np.sum(image_label_line==1)>0:        
        # 汽車車停車格
        parkingSpace_car, linears_formula_car, formula_line_parkline_main_car = fun_find_ParkingSpace(image_label_line,parking_points, 1)
        color_image = plot_parkingline(color_image, linears_formula_car, formula_line_parkline_main_car, (255,0,255))
    elif np.sum(image_label_line==6)>0:
        # 紅線
        # 用Y軸座標計算X軸座標
        red_line = line_traffic_formula[6]
        color_image = plot_line(color_image, red_line,  (0,0,255))
    


def fun_detection_TrafficViolation(img, bboxes, map_seg_label_line, map_seg_label_road):
    # 物件違規判斷
    # 依據物件偵測的結果進行判斷
    # img: original image
    # bboxes: detected objects' bboxes
    
    h_img, w_img = fun_getimageWH(img)

    # th_roi_h = [int(h_img/h_img)-1, h_img]
    # th_roi_w = [int(w_img/4), w_img]
    th_roi_h = [0, h_img]
    th_roi_w = [0, w_img]
    ROI_area = (max(th_roi_h)-min(th_roi_h))*(max(th_roi_w)-min(th_roi_w)) 
    
    img_result = img.copy()
    img_result = plot_ROI(
        img_result, [th_roi_w[0],th_roi_h[0],th_roi_w[1],th_roi_h[1]])
    
    #此function用來推論"線的方程式"，所以需要"線的segmentation"和"角點的segmentation"
    h_img, w_img = fun_getimageWH(map_seg_label_line)
    # image_label_line, mask_point
    # 2.1 ROI mask預測結果圖
    map_seg_label_line = fun_result_laneline_ROI_modify(map_seg_label_line)
    
    decision_boxes=[]
    for bbox in bboxes:
        object_label = bbox['label'][0]
        bbox_axis = bbox['points']
        
        # bbox左上右下的座標轉換成圖的長和寬
        bbox_x, bbox_y, bbox_w, bbox_h = fun_box2hw(bbox_axis)
        ### 刪除左下右下的carself
        #左上(Front鏡頭)排除
        if bbox_x < 500 and bbox_y > 800 :
            continue
        
        # 將detected的bbox拉大一點
        bbox_axis[0] = int(bbox_axis[0]-bbox_w*0.0)
        if bbox_axis[0]<=0: bbox_axis[0]=0
        bbox_axis[1] = int(bbox_axis[1]-bbox_h*0.1)
        if bbox_axis[1]<=0: bbox_axis[1]=0
        bbox_axis[2] = int(bbox_axis[2]+bbox_w*0.1)
        if bbox_axis[2]>=w_img: bbox_axis[2]=w_img-1
        if (object_label=='m'):
            bbox_axis[3] = int(bbox_axis[3]+bbox_h*0.1)
        if (object_label=='c'):
            bbox_axis[3] = int(bbox_axis[3]+bbox_h*0.2)
        if (object_label=='t'):
            bbox_axis[3] = int(bbox_axis[3]+bbox_h*0.2)
        if bbox_axis[3]>=h_img: bbox_axis[3]=h_img-1
        bbox_x, bbox_y, bbox_w, bbox_h = fun_box2hw(bbox_axis)
    
        bbox_h_range = [int(bbox_axis[1]),int(bbox_axis[3])]
        bbox_w_range = [int(bbox_axis[0]),int(bbox_axis[2])]
        box_area = (max(bbox_h_range)-min(bbox_h_range))*(max(bbox_w_range)-min(bbox_w_range))
        # object in ROI, object must in ROI will be classified.
        # 物件在ROI內才會進行判斷。
        if (((bbox_h_range[0]>= th_roi_h[0]) & (bbox_h_range[0] <= th_roi_h[1]))&\
            ((bbox_w_range[0]>= th_roi_w[0]) & (bbox_w_range[0] <= th_roi_w[1])))\
            &\
           (((bbox_h_range[1]>= th_roi_h[0]) & (bbox_h_range[1] <= th_roi_h[1]))&\
            ((bbox_w_range[1]>= th_roi_w[0]) & (bbox_w_range[1] <= th_roi_w[1]))):
            
            # 1. 車輛的bbox在ROI內的面積比例要大於10%才判斷
            # 2.機車在ROI內不須看比例
            if  ((box_area/ROI_area)>=0.1) & ((object_label=='b') | (object_label=='t') | (object_label=='c'))\
                | (object_label=='m'):
                decision_box = {}
                
                '''
                # 'sw':1,'dr':2,'ud':0;  map_seg_label_road: label_seg
                # 物件的框內，"可行駛道路segmentation"的結果
                # 我們要看此框的segmentation的結果左下角是不是人行道
                # 左下角是人行道的話是人行道違停的機率大。
                # 我們看框內左下(下10%和左50%的聯集區塊)的pixel數量
                '''
                box_map_seg_label_road  = map_seg_label_road[bbox_h_range[0]:bbox_h_range[1], bbox_w_range[0]:bbox_w_range[1]]
                pos_box_map_seg_label_road = np.where(box_map_seg_label_road==1)
                pos_box_map_seg_label_driven = np.where(box_map_seg_label_road==2)
                n_sw = np.sum((pos_box_map_seg_label_road[0]>(bbox_h*0.9))*(pos_box_map_seg_label_road[1]<(bbox_h*0.5))) # bbox_w
                n_da = np.sum((pos_box_map_seg_label_road[0]>(bbox_h*0.5)))
                '''
                # 物件的框內，"線的segmentation"結果
                算1. 機車停車格，2.汽車停車格，3. 紅線 的pixel數
                '''
                box_label_line = map_seg_label_line[bbox_h_range[0]:bbox_h_range[1], bbox_w_range[0]:bbox_w_range[1]]
                box_label_line_red = map_seg_label_line[bbox_h_range[0]+int(bbox_h_range[1]*0.5):bbox_h_range[1], bbox_w_range[0]:bbox_w_range[1]]
                n_motor_line = np.sum(box_label_line==0)  # 機車停車格       
                n_car_line = np.sum(box_label_line==1)  # 汽車停車格
                n_red_line = np.sum(box_label_line_red==6)  # 紅線
                n_bus_line = np.sum(box_label_line==2) #公車格
                n_white_line = np.sum(box_label_line==9)+np.sum(box_label_line==10)# 白線
                n_mstop_line = np.sum(box_label_line==3)  # 機車代轉區
                
                
                # 開始判斷準則
                if (n_sw>10) & (n_red_line<100):
                    '''
                    物件框的人行道pixel如果大於10個(預設可以調整)
                    且紅線小於100個(預設可以調整)，就是人行道違停
                    '''
                    decision_box['decision'] = 'Violation_parking_in_sidewalk'
                    decision_box['bbox'] = bbox
                    img_result = plot_bbox_Violation(img_result, bbox,(0,255,255))
                elif ( n_motor_line > n_red_line) & (n_motor_line > n_car_line) & (object_label=='m'):
                    '''
                    物件為機車
                    機車線停車線的pixel數量 > 紅線pixel數量
                    機車線停車線的pixel數量 > 汽車停車線的pixel數量
                    機車停在機車停車格，pass
                    '''
                    decision_box['decision'] = 'pass'
                    decision_box['bbox'] = bbox
                    img_result = plot_bbox(img_result, bbox)
                elif (n_car_line > n_red_line) & (n_car_line > n_motor_line) & (object_label=='c'):
                    '''
                    物件為汽車
                    汽車線停車線的pixel數量 > 紅線pixel數量
                    汽車線停車線的pixel數量 > 機車停車線的pixel數量
                    汽車停在汽車停車格，pass
                    '''
                    decision_box['decision'] = 'pass'
                    decision_box['bbox'] = bbox
                    img_result = plot_bbox(img_result, bbox)
                elif (((n_red_line > n_car_line) | (n_red_line > n_motor_line)) & (n_red_line>n_white_line)&(n_red_line > n_mstop_line)) & (n_red_line>=10) &\
                     ((object_label=='b') | (object_label=='t') | (object_label=='m') | (object_label=='c')):
                    '''
                    (紅線pixel數量 > 汽車線停車線的pixel數量) 或是 (紅線pixel數量 > 機車線停車線的pixel數量)
                    且
                    物件為b(大車)或t(卡車)或m(機車上無人)或是c(汽車)
                    就是紅線違停
                    '''
                    decision_box['decision'] = 'Violation_parking_in_redline'
                    decision_box['bbox'] = bbox
                    img_result = plot_bbox_Violation(img_result, bbox,(255,0,0))
                elif ((n_bus_line > n_car_line) & (n_bus_line > n_motor_line)) & (n_bus_line>=100) &\
                     ((object_label=='b') | (object_label=='t') | (object_label=='m') | (object_label=='c')):
                    '''
                    (紅線pixel數量 > 汽車線停車線的pixel數量) 或是 (紅線pixel數量 > 汽機車線停車線的pixel數量)
                    且
                    物件為b(大車)或t(卡車)或m(機車上無人)或是c(汽車)
                    就是公車格違停
                    '''
                    decision_box['decision'] = 'Violation_parking_in_redline'
                    decision_box['bbox'] = bbox
                    img_result = plot_bbox_Violation(img_result, bbox,(255,0,255))
                
                else:
                    '''
                    沒被規範到的預設為pass
                    '''
                    decision_box['decision'] = 'pass'
                    decision_box['bbox'] = bbox
                    img_result = plot_bbox(img_result, bbox)
                decision_boxes.append(decision_box)
            else:
                '''
                沒在ROI內的
                '''
                decision_box = {}
                decision_box['decision'] = 'pass'
                decision_box['bbox'] = bbox
                img_result = plot_bbox(img_result, bbox)        
                decision_boxes.append(decision_box)
    return decision_boxes, img_result
                
    
    
    
