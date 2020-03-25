# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:43:23 2019

@author: 07067
"""
import xml.etree.ElementTree as ET
xmlfile='32_114_TidingBlvd2-Gangqian-PEh_20190414_0600-2200_0014-1 (1).xml'
def parse_annotation_Elan_od_cvat(annotation_path, label_map):
    tree = ET.parse(xmlfile)
    roots = tree.getroot()
    object_bbox = {}
    for root in roots:    
        if root.tag =='track':
            object_id = root.attrib['id']
            label = root.attrib['label']
            for sub_root in root: 
                attribs = sub_root.attrib
                frame = attribs['frame']
                if frame not in object_bbox:
                    object_bbox[frame]={}
                if object_id not in object_bbox[frame]:
                    object_bbox[frame][object_id]={}
                object_bbox[frame][object_id]=attribs
                object_bbox[frame][object_id]['label']=label
    object_bboxes={}
    for k in sorted(object_bbox.keys()):
        boxes,labels,occluded=[],[],[]
        objects = object_bbox[k]
        for tmp_id in objects:
            tmp_object=objects[tmp_id]
            if tmp_object['outside']=='0':
                x_min = float(tmp_object['xtl'])
                y_min = float(tmp_object['ytl'])
                x_max = float(tmp_object['xbr'])
                y_max = float(tmp_object['ybr'])
                label = tmp_object['label']
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append([label])
                occluded.append(tmp_object['occluded'])
        object_bboxes[k]={'boxes': boxes, 'labels': labels, 'occluded': occluded}
    return object_bboxes




    
    
    
    
    
    