# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 10:25:56 2019

@author: 07067
"""
#plt.subplot(3,1,1)
#plt.imshow(argmax_feats_road)
#plt.subplot(3,1,2)
#plt.imshow(argmax_feats_lane)
#plt.subplot(3,1,3)
#plt.imshow(frame_np_img)

# labels_line={'機車格':0,
#              '汽車格':1,
#              '公車格':2,
#              '機車等待區':3,'機車待轉區':3,
#              '斑馬線':4,
#              '網黃線':5,
#              '紅線':6,
#              '黃線':7,'黃線':7,
#              '雙黃線':8,'雙黃線':8,
#              '白線 實線':9,
#              '白線 虛線':10}
#              'Background':11}

# road segmentation
# pallete = [ [0, 0, 0], [255, 0, 0], [0, 255, 0] ]
# labels_seg={'人行道':1,
#             '可行駛道路區域':2,
#             'Ignore':0}
import numpy as np
def fun_intergate_seg_LaneandRoad(argmax_feats_lane, argmax_feats_road):
    # 將紅線部分如果不在非路上則修成背景。
    pos_redline=np.where(argmax_feats_lane==7)
    pos=np.where(argmax_feats_road[pos_redline[0],pos_redline[1]]==0)[0]
    argmax_feats_lane[pos_redline[0][pos],pos_redline[1][pos]]=100
    return argmax_feats_lane,argmax_feats_road


