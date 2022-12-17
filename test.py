#!~/Desktop/myFolder/Grad project/torch/pytorch-env/bin/python
import cv2
import torch
import numpy as np
import json
from ..detectron2.config import get_cfg
from ..detectron2.data.detection_utils import read_image
from ..detectron2.engine.defaults import DefaultPredictor


def detect_human(image_path):
    confidence_threshold = 0.7
    config_file = 'detect_body_parts/extract/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
    opts = ['MODEL.WEIGHTS',
            'detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl',
            'MODEL.DEVICE', 'cpu']

    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS='model_final_f10217.pkl'
    # cfg.merge_from_list(opts)
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    cfg.freeze()

    img = read_image(image_path, format="BGR")
    image = cv2.imread(image_path)

    predictor = DefaultPredictor(cfg)
    predictions = predictor(img)
    # print(predictions)

    i = 0
    for box in predictions["instances"].pred_boxes:
        if predictions["instances"].pred_classes[i] == 0:

            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            pred_mask = predictions["instances"].pred_masks[i]
            mask = np.zeros(image.shape, dtype=np.uint8)

            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    if pred_mask[i][j]:
                        mask[i][j] = 255

            result = cv2.bitwise_and(image, mask)
            result[mask == 0] = 255
            (startX, startY, endX, endY) = box
            startX = startX.type(torch.int64)
            startY = startY.type(torch.int64)
            endY = endY.type(torch.int64)
            endX = endX.type(torch.int64)
            crop_img = result[startY:endY, startX:endX + 5]
            # cv2.imshow('image' + str(startX), crop_img)

        i = +1
    return result, crop_img.shape[0]

def detect_joints(image_path):
    confidence_threshold = 0.7
    config_file = 'detect_body_parts/extract/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml'
    opts = ['MODEL.WEIGHTS',
            'detectron2://COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl',
            'MODEL.DEVICE', 'cpu']
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS='model_final_a6e10b.pkl'
    # cfg.merge_from_list(opts)
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    cfg.freeze()

    img = read_image(image_path, format="BGR")
    predictor = DefaultPredictor(cfg)
    predictions = predictor(img)
    return predictions
def get_chest_width(predictions,human_img):

    keypoints = predictions["instances"].pred_keypoints[0]
    for idx, keypoint in enumerate(keypoints):
        x, y, prob = keypoint
        if prob > 0.05:
            if idx == 5:
                endY = y

    endY = endY.type(torch.int64)
    count_pixels = 0
    for i in range(0, human_img.shape[1]):
        pixel = human_img[endY, i]
        if not all(pixel == [255, 255, 255]):
            count_pixels=count_pixels+1

    return count_pixels


def get_back_width(predictions,human_img):

    keypoints = predictions["instances"].pred_keypoints[0]
    for idx, keypoint in enumerate(keypoints):
        x, y, prob = keypoint
        if prob > 0.05:
            if idx == 5:
                endY = y

    endY = endY.type(torch.int64)
    count_pixels = 0
    for i in range(0, human_img.shape[1]):
        pixel = human_img[endY, i]
        if not all(pixel == [255, 255, 255]):
            count_pixels=count_pixels+1

    return count_pixels


def get_back_depth(predictions,human_img):

    keypoints = predictions["instances"].pred_keypoints[0]
    for idx, keypoint in enumerate(keypoints):
        x, y, prob = keypoint
        if prob > 0.05:
            if idx == 5 or idx == 6:
                endY = y

    endY = endY.type(torch.int64)
    count_pixels = 0
    for i in range(0, human_img.shape[1]):
        pixel = human_img[endY, i]
        if not all(pixel == [255, 255, 255]):
            count_pixels=count_pixels+1

    return count_pixels


def get_chest_depth(predictions,human_img):

    keypoints = predictions["instances"].pred_keypoints[0]
    for idx, keypoint in enumerate(keypoints):
        x, y, prob = keypoint
        if prob > 0.05:
            if idx == 5 or idx == 6:
                endY = y

    endY = endY.type(torch.int64)
    count_pixels = 0
    for i in range(0, human_img.shape[1]):
        pixel = human_img[endY, i]
        if not all(pixel == [255, 255, 255]):
            count_pixels=count_pixels+1

    return count_pixels

def get_hip_width_front(predictions,human_img):

    keypoints = predictions["instances"].pred_keypoints[0]
    for idx, keypoint in enumerate(keypoints):
        x, y, prob = keypoint
        if prob > 0.05:
            if idx == 11:
                endY = y

    endY = endY.type(torch.int64)
    count_pixels = 0
    for i in range(0, human_img.shape[1]):
        pixel = human_img[endY, i]
        if not all(pixel == [255, 255, 255]):
            count_pixels=count_pixels+1

    return count_pixels

def get_hip_width_side(predictions,human_img):

    keypoints = predictions["instances"].pred_keypoints[0]
    for idx, keypoint in enumerate(keypoints):
        x, y, prob = keypoint
        if prob > 0.05:
            if idx == 11 or idx == 12:
                endY = y

    endY = endY.type(torch.int64)
    count_pixels = 0
    for i in range(0, human_img.shape[1]):
        pixel = human_img[endY, i]
        if not all(pixel == [255, 255, 255]):
            count_pixels=count_pixels+1

    return count_pixels
def calculate(a,b):
    return (22/7) * (a + b) * (1 +( (3 * ((a - b) ** 2)) / ((10 + ((4 - ((3 * ((a - b) ** 2)) / ((a + b) ** 2)) ) ** 0.5)) * ((a + b) ** 2))))
def getMeasurements(id,real_height,frontImagePath,sideImagePath,backImagePath):
    #chest
    front_human_img, height = detect_human(frontImagePath)
    pixels_per_metric = float(height) / float(real_height)
    predictions = detect_joints(frontImagePath)

    count_chest_pixels = get_chest_width(predictions,front_human_img)
    real_chest_width = float(count_chest_pixels) / float(pixels_per_metric)\
    #Back
    pixels_per_metric = float(height) / float(real_height)
    predictions = detect_joints(backImagePath)

    count_back_pixels = get_back_width(predictions,front_human_img)
    real_back_width = float(count_back_pixels) / float(pixels_per_metric)
#Hip
    count_hip_pixels = get_hip_width_front(predictions,front_human_img)
    real_hip_width = (float(count_hip_pixels) / float(pixels_per_metric))

    side_human_img, height = detect_human(sideImagePath)
    pixels_per_metric = float(height) / float(real_height)
    predictions = detect_joints(sideImagePath)
    
#Hip Side Depth
    count_hip_side_pixels = get_hip_width_side(predictions,side_human_img)
    real_hip_width_side = (float(count_hip_side_pixels) / float(pixels_per_metric))
    
 #Chest Side Depth   
    count_chest_side_pixels = get_chest_depth(predictions,side_human_img)
    real_chest_depth_side = (float(count_chest_side_pixels) / float(pixels_per_metric))
 #Back Side Depth   
    count_back_side_pixels = get_back_depth(predictions,side_human_img)
    real_back_depth_side = (float(count_back_side_pixels) / float(pixels_per_metric))

    
    hip=calculate(float(real_hip_width) / 2, float(real_hip_width_side) / 2)
    chest=calculate(float(real_chest_width) / 2, float(real_chest_depth_side) / 2)
    back=calculate(float(real_back_width) / 2, float(real_back_depth_side) / 2)
    measurements={'hip':hip,'chest':chest,'back':back}
    with open('data_'+str(id)+'.txt', 'w') as outfile:
        json.dump(measurements, outfile)
