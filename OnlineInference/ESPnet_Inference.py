import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
import glob
import cv2
import Model as Net
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from argparse import ArgumentParser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pallete_line = [ [255, 0, 255], [128, 0, 128], [0, 64, 64], [0, 0, 0], [0, 0, 0], \
            [255, 0, 0], [0, 255, 0], [255, 0, 0], [0, 0, 255], [255, 255, 0], \
            [255, 0, 255], [0,0,0] ]
pallete_road= [ [0,0,0], [255, 255, 0], [0, 255, 0]]

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

def fun_normalimage_pytorchformat(original_image):
    # gloabl mean and std values
    mean = [124.29382, 123.20624, 122.044754]
    std = [59.38708, 60.81311, 57.652054]
    img = np.array(original_image).astype(np.float32)
    for j in range(3):
        img[:, :, j] -= mean[j]
    for j in range(3):
        img[:, :, j] /= std[j]
    img /= 255
    return img

def evaluateModel(model, original_image, inWidth, inHeight, flag_road):
    '''
    original_image: PIl image
    '''
    original_image = np.array(original_image)
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
        pallete = pallete_line
    classMap_np_output_color = np.zeros((ori_h, ori_w, ori_ch), dtype=np.uint8)
    for idx in range(len(pallete)):
        [r, g, b] = pallete[idx]
        classMap_np_output_color[classMap_np_output == idx,:] = [b, g, r]

    return classMap_np_output, classMap_np_output_color


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
