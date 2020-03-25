import os
from src.utils_car import *
from PIL import Image, ImageDraw, ImageFont
import glob
import torch
import numpy as np
import time

# Load model checkpoint
# checkpoint = '/home/ahan/Project/Segmentaion_Model/Singapore_Model_Test/models/od_NoPretrain/BEST_checkpoint.pth.tar'

# checkpoint_path = checkpoint
# checkpoint = torch.load(checkpoint)
# start_epoch = checkpoint['epoch'] + 1
# best_loss = checkpoint['best_loss']
# print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, best_loss))
# model = checkpoint['model']
# model = model.to(device)
# model.eval()

# dummy_input = torch.randn(1, 3, 352, 352)
# torch.onnx.export(model, dummy_input.to(device), r'/home/ahan/Desktop/ELANetV3_2/ElanetV3_2_od.onnx')

# Transforms
preprocess_transforms = transforms.Compose([
    transforms.Resize((352, 352)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ]
)

def detect(model, original_image, min_score, max_overlap, top_k, device):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """
    org_image = original_image.copy()
    # Transform
    #image = normalize(to_tensor(resize(original_image)))
    image = preprocess_transforms(org_image)
    # image = transforms.functional.to_pil_image(image)
    # return image, None
    # Move to default device
    image = image.to(device)

    
    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # for object detection (only) model
    # predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    with torch.no_grad():
        det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)
    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')
    det_scores = det_scores[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [org_image.width, org_image.height, org_image.width, org_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [Elan_od_singapore_rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD_model.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        det_boxes_ = []
        return original_image, det_boxes_

    # Annotate
    annotated_image=[]
    # annotated_image = org_image
    # draw = ImageDraw.Draw(annotated_image)
    # # font = ImageFont.truetype("./calibril.ttf", 15)
    # font = ImageFont.load_default().font

    # Suppress specific classes, if needed
    # for i in range(det_boxes.size(0)):
    #     if suppress is not None:
    #         if det_labels[i] in suppress:
    #             continue
    #
    #     # Boxes
    #     box_location = det_boxes[i].tolist()
    #     draw.rectangle(xy=box_location, outline=Elan_od_singapore_label_color_map[det_labels[i]])
    #     draw.rectangle(xy=[l + 1. for l in box_location], outline=Elan_od_singapore_label_color_map[
    #         det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
    #     # Text
    #     text_size = font.getsize(det_labels[i].upper())
    #     text_location = [box_location[0] + 3., box_location[1] - text_size[1]]
    #     textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
    #                         box_location[1]]
    #     draw.rectangle(xy=textbox_location, fill=Elan_od_singapore_label_color_map[det_labels[i]])
    #     draw.text(xy=text_location, text=det_labels[i].lower(), fill='white',
    #               font=font)
    # del draw
    det_boxes = np.array(det_boxes.numpy())
    det_scores = np.array(det_scores.numpy())
    bboxes = [ {'label': tmp_label, 'points': tmp_box, 'score': tmp_score} for tmp_box, tmp_label, tmp_score in zip(det_boxes, det_labels, det_scores)]   
    
    return annotated_image, bboxes

if __name__ == '__main__':

    img_path_list = glob.glob("/mnt/83a7cab6-2970-47cf-b4ae-9e770da2cb65/dataset/Elan/Singapore_ViolationDetection/OV_001-1-Segmentation/*.jpg")
    img_path_list.sort()

    for frmaeidx, img_path in enumerate(img_path_list):
        # if frmaeidx < 300:
        #     continue
        if not os.path.exists("/home/ahan/Project/Segmentaion_Model/Singapore_Model_Test/demo"):
            os.makedirs("/home/ahan/Project/Segmentaion_Model/Singapore_Model_Test/demo")
        print(img_path.split("/")[-1])
        original_image = Image.open(img_path, mode='r')
        original_image = original_image.convert('RGB')


        annotated_image, detected_boxes = detect(original_image, min_score=0.4, max_overlap=0.5, top_k=200)
        annotated_image_ = cv2.cvtColor(np.asarray(annotated_image),cv2.COLOR_RGB2BGR)
        cv2.imshow('img', annotated_image_ )
        key = cv2.waitKey(100)
        if key in [ord('q') or ord('Q')]:
            break
        # for box in detected_boxes:
        #     print(box)

        # annotated_image.save("demo/{}.png".format(img_path.split("/")[-1].split(".")[0]))
        
