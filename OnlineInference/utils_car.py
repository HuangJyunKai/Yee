import json
import os
import torch
import random
import numpy as np
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as FT
from torchvision import transforms
from PIL import Image
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# SINGAPORE_od labels and color setting (Bus)
########################################################################################################################
# Label map
Elan_od_singapore_rev_label_map={}
Elan_od_singapore_labels = ('b', 't', 'm', 'k', 'c')
Elan_od_singapore_label_map = {k: v + 1 for v, k in enumerate(Elan_od_singapore_labels)}
Elan_od_singapore_label_map['background'] = 0
Elan_od_singapore_rev_label_map = {v: k for k, v in Elan_od_singapore_label_map.items()}  # Inverse mapping

# Color map for bounding boxes of detected objects from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
Elan_od_singapore_distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#008080']
Elan_od_singapore_label_color_map = {k: Elan_od_singapore_distinct_colors[i] for i, k in enumerate(Elan_od_singapore_label_map.keys())}
########################################################################################################################


def parse_annotation_SINGAPORE_od(annotation_path, label_map):
    
    bboxes = fun_json2ObjLoc(annotation_path)
    
    boxes = list()
    labels = list()
    difficulties = list()
    # difficulties = [0]*len(bboxes)
    
    for box in bboxes:

        # difficult = int(object.find('difficult').text == '1')
        difficult = int(0)

        label = box['label']
        if len(label) > 1:
            label = str(label[0])
        if label not in label_map:
            print('False Label Image: {}'.format(annotation_path))
            print(box['label'])
            continue
        box_pp = box['points']

        boxes.append(box_pp)
        labels.append(label_map[label])
        difficulties.append(difficult)
    return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}


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
            box['points']=[x_min,y_min,x_max, y_max]
            bboxes.append(box)
    return bboxes

def create_data_lists_SINGAPORE_od(train_path, test_path, output_folder, flag_apply= 'Day'):
    """
    Create lists of images, the bounding boxes and labels of the objects in these images, and save these to file.

    :param train_path: path to the 'train.txt' folder
    :param test_path: path to the 'test.txt' folder
    :param output_folder: folder where the JSONs must be saved
    """

    label_map = Elan_od_singapore_label_map
    # if flag_apply == 'Day':
    #     label_map = Elan_od_CVAT_Day_label_map
    # if flag_apply == 'Night':
    #     label_map = Elan_od_CVAT_Night_label_map

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if isinstance(train_path, str):
        train_path = [os.path.abspath(train_path)]
    elif isinstance(train_path, list):
        train_path = [os.path.abspath(x) for x in train_path]

    if isinstance(test_path, str):
        test_path = [os.path.abspath(test_path)]
    elif isinstance(test_path, list):
        test_path = [os.path.abspath(x) for x in test_path]
    
    print(train_path)
    print(test_path)

    train_images = list()
    train_objects = list()
    n_objects = 0

    # Training data
    for path in train_path:

        # Find IDs of images in training data
        with open(path) as f:
            ids = f.read().splitlines()

        for id in ids:
            id = id.split(",")
            train_img = id[0].strip()
            train_label = id[1].strip()
            # print(train_img)
            # print(train_label)
            
            # Parse annotation's XML file
            objects = parse_annotation_SINGAPORE_od(train_label, label_map)
            
            if len(objects['boxes']) == 0:
                continue
            n_objects += len(objects['boxes'])
            train_images.append(train_img)
            train_objects.append(objects)

    assert len(train_objects) == len(train_images)

    # Save to file
    with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j:
        json.dump(train_images, j)
    with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
        json.dump(train_objects, j)
    with open(os.path.join(output_folder, 'label_map.json'), 'w') as j:
        json.dump(label_map, j)  # save label map too

    print('\nThere are %d training images containing a total of %d objects. Files have been saved to %s.' % (
        len(train_images), n_objects, os.path.abspath(output_folder)))

    # # Validation data
    # test_images = list()
    # test_objects = list()
    # n_objects = 0

    # for path in test_path:
    #     # Find IDs of images in validation data
    #     with open(path) as f:
    #         ids = f.read().splitlines()

    #     for id in ids:
    #         id = id.split(",")
    #         test_img = id[0].strip()
    #         test_label = id[1].strip()

    #         # Parse annotation's XML file
    #         objects = parse_annotation_Elan_od(test_label,label_map)
    #         if len(objects['boxes']) == 0:
    #             continue
    #         n_objects += len(objects['boxes'])
    #         test_images.append(test_img)
    #         test_objects.append(objects)

    # assert len(test_objects) == len(test_images)

    # # Save to file
    # with open(os.path.join(output_folder, 'TEST_images.json'), 'w') as j:
    #     json.dump(test_images, j)
    # with open(os.path.join(output_folder, 'TEST_objects.json'), 'w') as j:
    #     json.dump(test_objects, j)

    # print('\nThere are %d validation images containing a total of %d objects. Files have been saved to %s.' % (
    #     len(test_images), n_objects, os.path.abspath(output_folder)))

	
def decimate(tensor, m):
    """
    Decimate a tensor by a factor 'm', i.e. downsample by keeping every 'm'th value.

    This is used when we convert FC layers to equivalent Convolutional layers, BUT of a smaller size.

    :param tensor: tensor to be decimated
    :param m: list of decimation factors for each dimension of the tensor; None if not to be decimated along a dimension
    :return: decimated tensor
    """
    assert tensor.dim() == len(m)
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(dim=d,
                                         index=torch.arange(start=0, end=tensor.size(d), step=m[d]).long())

    return tensor



def xy_to_cxcy(xy):
    """
    Convert bounding boxes from boundary coordinates (x_min, y_min, x_max, y_max) to center-size coordinates (c_x, c_y, w, h).

    :param xy: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h


def cxcy_to_xy(cxcy):
    """
    Convert bounding boxes from center-size coordinates (c_x, c_y, w, h) to boundary coordinates (x_min, y_min, x_max, y_max).

    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)  # x_max, y_max


def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    """
    Encode bounding boxes (that are in center-size form) w.r.t. the corresponding prior boxes (that are in center-size form).

    For the center coordinates, find the offset with respect to the prior box, and scale by the size of the prior box.
    For the size coordinates, scale by the size of the prior box, and convert to the log-space.

    In the model, we are predicting bounding box coordinates in this encoded form.

    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding must be performed, a tensor of size (n_priors, 4)
    :return: encoded bounding boxes, a tensor of size (n_priors, 4)
    """

    # The 10 and 5 below are referred to as 'variances' in the original Caffe repo, completely empirical
    # They are for some sort of numerical conditioning, for 'scaling the localization gradient'
    # See https://github.com/weiliu89/caffe/issues/155
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)  # g_w, g_h


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    """
    Decode bounding box coordinates predicted by the model, since they are encoded in the form mentioned above.

    They are decoded into center-size coordinates.

    This is the inverse of the function above.

    :param gcxgcy: encoded bounding boxes, i.e. output of the model, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding is defined, a tensor of size (n_priors, 4)
    :return: decoded bounding boxes in center-size form, a tensor of size (n_priors, 4)
    """

    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
                      torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)  # w, h


def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    # #box iou
    # output = intersection/ areas_set_2

    return intersection / union  # (n1, n2)



def load_best_checkpoint(model, save_path):
    """
    Load the best model checkpoint.

    :param model: model
    :param save_path: the path the saved the best model checkpoint
    """
    filename = 'checkpoint.pth.tar'
    checkpoint = torch.load(os.path.join(save_path, 'BEST_' + filename))
    best_model = checkpoint['model']
    model.load_state_dict(best_model.state_dict())
    print("loaded the model weight from {}".format(os.path.join(save_path, 'BEST_' + filename)))
