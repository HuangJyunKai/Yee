import numpy as np
import torch
import glob
import cv2
import src.models.Models as Net
import os
import time 
import threading
from queue import Queue


# R, G, B
pallete_lane = [ [0, 0, 255],
                [102, 0, 204],
                [0, 204, 0], 
                [0, 255, 255], 
                [192, 192, 192],
                [255, 128, 0], 
                [255, 0, 0], 
                [255, 255, 0], 
                [220, 220, 0], 
                [255, 255, 255],
                [255, 255, 255],
                [0,0,0]]
pallete_road= [ [0,0,0], [255, 255, 0], [0, 255, 0]]

# labels_line={'機車格':0, 藍色
#              '汽車格':1, 紫色
#              '公車格':2, 綠色
#              '機車等待區':3,'機車待轉區':3, 水藍色
#              '斑馬線':4,                 灰色
#              '網黃線':5,                 橘色
#              '紅線':6,                   紅色
#              '黃線':7,'黃線':7,          黃色
#              '雙黃線':8,'雙黃線':8,      暗黃色
#              '白線 實線':9,
#              '白線 虛線':10}
#              'Background':11}

# road segmentation
# pallete = [ [0, 0, 0], [255, 0, 0], [0, 255, 0] ]
# labels_seg={'人行道':1,
#             '可行駛道路區域':2,
#             'Ignore':0}


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

def netParams(model):
    '''
    helper function to see total network parameters
    :param model: model
    :return: total network parameters
    '''
    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p

    return total_paramters

def fun_normalimage_pytorchformat(img):
    # gloabl mean and std values
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img /= 255
    for j in range(3):
        img[:, :, j] = (img[:, :, j]-mean[j])/std[j]

    
    return img

def evaluateModel(model, original_image, inWidth, inHeight, flag_road, device):
    '''
    original_image: PIl image
    '''
    original_image = np.array(original_image).astype(np.float32)
    ori_h = original_image.shape[0]
    ori_w = original_image.shape[1]
    ori_ch = original_image.shape[2]
    # original RGB image 
    # 1. normalization (z-score)
    image_4_seg = fun_normalimage_pytorchformat(original_image)
    # resize the image
    image_4_seg = cv2.resize(image_4_seg, (inWidth, inHeight))

    image_4_seg = image_4_seg.transpose((2, 0, 1))

    img_tensor = torch.FloatTensor((image_4_seg))
    img_tensor = torch.unsqueeze(img_tensor, 0)  # add a batch dimension
    # img_variable = Variable(img_tensor, volatile=True)
#    img_variable = Variable(img_tensor)
    # if args.gpu:
    img_variable = img_tensor.to(device)
    img_out = model(img_variable)

    classMap_np_output = np.array(img_out[0].max(0)[1].byte().cpu().data.numpy())
    classMap_np_output = cv2.resize(classMap_np_output, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
    # if args.colored:
    if flag_road==1:
        pallete = pallete_road
    else:
        pallete = pallete_lane
    classMap_np_output_color = np.zeros((ori_h, ori_w, ori_ch), dtype=np.uint8)
    for idx in range(len(pallete)):
        [r, g, b] = pallete[idx]
        classMap_np_output_color[classMap_np_output == idx,:] = [b, g, r]

    return classMap_np_output, classMap_np_output_color


def thread_seg_model_road(model_road, img_variable, q_sed_road):
    img_out = model_road(img_variable)
    q_sed_road.put(img_out)
def thread_seg_model_lane(model_lane, img_variable, q_sed_lane):
    img_out = model_lane(img_variable)
    q_sed_lane.put(img_out)



def evaluateModel_models(model_road, model_lane, original_image, inWidth, inHeight, device):
    '''
    original_image: PIl image
    '''
    q_sed_road = Queue()   # 宣告 Queue 物件
    q_sed_lane = Queue()   # 宣告 Queue 物件
    original_image = np.array(original_image).astype(np.float32)
    ori_h = original_image.shape[0]
    ori_w = original_image.shape[1]
    ori_ch = original_image.shape[2]
    # original RGB image 
    # 1. normalization (z-score)
    image_4_seg = fun_normalimage_pytorchformat(original_image)
    # resize the image
    image_4_seg = cv2.resize(image_4_seg, (inWidth, inHeight))

    image_4_seg = image_4_seg.transpose((2, 0, 1))

    img_tensor = torch.FloatTensor(image_4_seg)
    img_tensor = torch.unsqueeze(img_tensor, 0)  # add a batch dimension
    # if args.gpu:
    img_variable = img_tensor.to(device)
    

    # time_start = time.time()
    t1 = threading.Thread(target = thread_seg_model_road, args=(model_road, img_variable, q_sed_road))
    t2 = threading.Thread(target = thread_seg_model_lane, args=(model_lane, img_variable, q_sed_lane))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    # print('seg model inference time:{}'.format(time.time()-time_start))
    
    img_out_road = q_sed_road.get()
    img_out_lane = q_sed_lane.get()
    
    # time_start = time.time()
    classMap_np_output_road = np.array(img_out_road[0].max(0)[1].byte().cpu().data.numpy())
    classMap_np_output_lane = np.array(img_out_lane[0].max(0)[1].byte().cpu().data.numpy())
    ################################################# relabel for kevin's line seg #########################################################
    classMap_np_output_lane[classMap_np_output_lane==0]=100
    classMap_np_output_lane -= 1
    classMap_np_output_lane[classMap_np_output_lane==99]=11
    ################################################# relabel for kevin's line seg #########################################################
    # closing
    classMap_np_output_lane = cv2.morphologyEx(classMap_np_output_lane, cv2.MORPH_CLOSE, kernel, iterations=1)
    classMap_np_output_road = cv2.morphologyEx(classMap_np_output_road, cv2.MORPH_CLOSE, kernel, iterations=1)
    # 假設紅線不在可行駛道路上(可能為誤判)，則修正成背景
    # classMap_np_output_road: 2 = 可行駛道路
    # classMap_np_output_lane: 6 = 紅線
    pos_notredline = np.where((classMap_np_output_road!=2) * (classMap_np_output_lane==6))
    if len(pos_notredline[0])!=0:
        classMap_np_output_lane[pos_notredline[0],pos_notredline[1]]=11

    classMap_np_output_lane = cv2.resize(classMap_np_output_lane, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
    classMap_np_output_road = cv2.resize(classMap_np_output_road, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)

    # print('seg image posterior time label map:{}'.format(time.time()-time_start))
    
    time_start = time.time()

    classMap_np_output_color_road  = np.zeros((ori_h, ori_w, ori_ch), dtype=np.uint8)
    classMap_np_output_color_lane  = np.zeros((ori_h, ori_w, ori_ch), dtype=np.uint8)
    for idx in range(len(pallete_road)):
        [r, g, b] = pallete_road[idx]
#        classMap_np_output_color_road[classMap_np_output_road == idx,:] = [b, g, r]
        classMap_np_output_color_road[classMap_np_output_road == idx,:] = [r,g,b]
    for idx in range(len(pallete_lane)):
        [r, g, b] = pallete_lane[idx]
#        classMap_np_output_color_lane[classMap_np_output_lane == idx,:] = [b, g, r]
        classMap_np_output_color_lane[classMap_np_output_lane == idx,:] = [r,g,b]
    print('seg image posterior time color map:{}'.format(time.time()-time_start))

    return classMap_np_output_road, classMap_np_output_lane, classMap_np_output_color_road ,classMap_np_output_color_lane


#
## load lane segmentation model
#model_weight_file = r'/home/ahan/Project/Segmentaion_Model/Singapore_Model_Test/models/model_300_lane_512_256_p2_q3.pth'
#if not os.path.isfile(model_weight_file):
#    print('Pre-trained model file does not exist !!!!!')
#model_lane_seg = Net.ESPNet(classes=12, p=2, q=3)  # Net.Mobile_SegNetDilatedIA_C_stage1(20)
#model_lane_seg.load_state_dict(torch.load(model_weight_file))
#model_lane_seg = model_lane_seg.to(device)
#model_lane_seg.eval()
#
## load road segmentation model
#model_weight_file = r'/home/ahan/Project/Segmentaion_Model/Singapore_Model_Test/models/model_213_road_512_256_p2_q3.pth'
#if not os.path.isfile(model_weight_file):
#    print('Pre-trained model file does not exist !!!!!')
#model_road_seg = Net.ESPNet_corner_heatmap(classes=3, p=2, q=3)  # Net.Mobile_SegNetDilatedIA_C_stage1(20)
#model_road_seg.load_state_dict(torch.load(model_weight_file))
#model_road_seg = model_road_seg.to(device)
#model_road_seg.eval()
#
## make onnx model
#if False: 
#    x = Variable(torch.randn(1, 3, 256, 512))
#    x = x.to(device)
#    print("x.shape", x.shape)
#    torch.onnx.export(model_lane_seg, x, './model_lane_seg.onnx', export_params=True, verbose=False)
#    print("netParams(model_lane_seg)", netParams(model_lane_seg))
#
#    x = Variable(torch.randn(1, 3, 256, 512))
#    x = x.to(device)
#    print("x.shape", x.shape)
#    torch.onnx.export(model_road_seg, x, './model_road_seg.onnx', export_params=True, verbose=False)
#    print("netParams(model_road_seg)", netParams(model_road_seg))

#def main(args):
#    # read all the images in the folder
#    image_list = glob.glob(args.data_dir + os.sep + '*.jpg')
#    image_list.sort()
#
#    if not os.path.isdir(args.savedir):
#        os.mkdir(args.savedir)
#
#    # videoWriter = cv2.VideoWriter('./result_U_512_256_p2_q3_1118_focal.avi', cv2.VideoWriter_fourcc(*'XVID'), 15, (args.inWidth, args.inHeight))
#    # if not videoWriter.isOpened():
#    #     print('Video writer failed !!!!')
#    #     assert False
#
#    for idx, img_name in enumerate(image_list):
#        if idx % 100 == 0:
#            print(idx)
#
#        # result_img, out_feas, argmax_feats = evaluateModel(args, model_lane_seg, img_name, inWidth=512, inHeight=256)
#        result_img, out_feas, argmax_feats = evaluateModel(model_road_seg, img_name, inWidth=512, inHeight=256)
#
#        # videoWriter.write(result_img)
#        cv2.imshow('img', result_img)
#        key = cv2.waitKey()
#        if key in [ ord('q') or ord('Q') ]:
#            break
#    
#    # videoWriter.release()
#
#if __name__ == '__main__':
#    parser = ArgumentParser()
#    # parser.add_argument('--model', default="ESPNet", help='Model name')
#    parser.add_argument('--data_dir', default="./data/testing", help='Data directory')
#    # parser.add_argument('--inWidth', type=int, default=512, help='Width of RGB image')
#    # parser.add_argument('--inHeight', type=int, default=256, help='Height of RGB image')
#    parser.add_argument('--savedir', default='./results', help='directory to save the results')
#    # parser.add_argument('--gpu', default=True, type=bool, help='Run on CPU or GPU. If TRUE, then GPU.')
#    # parser.add_argument('--colored', default=True, type=bool, help='If you want to visualize the '
#    #                                                                'segmentation masks in color')
#    # parser.add_argument('--overlay', default=True, type=bool, help='If you want to visualize the '
#    #                                                                'segmentation masks overlayed on top of RGB image')
#
#    args = parser.parse_args()
#
#    # if args.overlay:
#    #     args.colored = True # This has to be true if you want to overlay
#    main(args)
