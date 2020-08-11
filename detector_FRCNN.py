'''
Implement and test car detection (localization)
'''

import numpy as np
# import tensorflow as tf
import torch
import torchvision
import torchvision.transforms as T
import cv2
from PIL import Image
import os
from matplotlib import pyplot as plt
import time
from glob import glob
cwd = os.path.dirname(os.path.realpath(__file__))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device)
model.eval()
#
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
W_orig = 1280
H_orig = 720

class CarDetector(object):
    def __init__(self):
        #
        self.car_boxes = []
        os.chdir(cwd)


    # Helper function to convert image into numpy array
    def load_image_into_numpy_array(self, image):
         (im_width, im_height) = image.size
         return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    def get_localization(self, image):

        """Determines the locations of the cars in the image

        Args:
            image: camera image

        Returns:
            orig list of bounding boxes: coordinates [y_up, x_left, y_down, x_right]
            revised list of bounding boxes:  coord [cx, cy, width, height]

        """
        # initialize the returned list
        self.car_boxes = []
        # open the input image
        img = image
        #  put the image into tensor form; start by creating a transformation
        transform = T.Compose([T.ToTensor()])
        # MES - perform the transformation and send to GPU (if used)
        img = transform(img).to(device)
        #  use the NN model to generate object detection prediction
        pred = model([img])
        #  get the confidence score - confidence that the object label applied is correct
        pred_score = list(pred[0]['scores'].detach().cpu().numpy())
        #  get the COCO class name for each predicted object label
        pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
        #  get the bounding box for each predicted object detection
        #  from model, boxes are in (x1, y1), (x2, y2) format
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]

        #  find the indices of the detections with scores above threshold
        score_threshold = float(0.5)        # minimum score to filter weak detections
        pred_t = [pred_score.index(x) for x in pred_score if x>score_threshold]

        # # debug
        # print(" ")
        # print("In get_prediction:")
        # print("img_path: ", img_path)
        # print("pred_score: ", pred_score)
        # print("pred_class: ", pred_class)
        # print("pred_boxes: ", pred_boxes)
        # print("pred_t: ", pred_t)

        #  remove the predictions with confidence score below the prescribed threshold
        #   also remove the detections for objects outside the object classes of interest
        list_clsid_keep = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']
        thresh_score = []
        thresh_boxes = []
        thresh_class = []
        for idx in pred_t:
            if pred_class[idx] in list_clsid_keep:
                thresh_score.append(pred_score[idx])
                thresh_boxes.append(pred_boxes[idx])
                thresh_class.append(pred_class[idx])
            #
        #
        # do the NMA suppression only if there is at least one detection:
        nms_scores = []
        nms_boxes = []
        nms_pred_cls = []
        if len(thresh_score) > 0:
            # apply non maxima suppression to reduce duplicate boxes
            nms_threshold = float(0.3)         # threshold when applying non-maxima suppression (IOU minimum at which suppression starts?)
            # since NMSBoxes requires boxes in [left, top, width, height] format, re-format the boxes
            boxes_rev = []
            scores_rev = []
            box_metrics = []
            for j, box in enumerate(thresh_boxes):
                (x1, y1), (x2, y2) = box
                left = int(x1)
                top = int(y1)
                width = int(abs(x2 - x1))
                height = int(abs(y2 - y1))
                area = width * height
                as_ratio = float(height) / float(width)
                cx = int(abs(x2 + x1)/2.0)
                cy = int(abs(y2 + y1)/2.0)
                box_metrics.append([cx, cy, width, height, area, as_ratio])
                boxes_rev.append([left, top, width, height])
                scores_rev.append(float(thresh_score[j]))
            #  make call to NMSBoxes
            idxs = cv2.dnn.NMSBoxes(boxes_rev, scores_rev, score_threshold, nms_threshold)

            # convert numpy array of indices to list
            list1_idxs = list(idxs)
            list2_idxs = []
            for element in list1_idxs:
                list2_idxs.append(int(element[0]))

            # # debug
            # print("idxs: ", idxs)
            # print("type of idxs: ", type(idxs))
            # print("list1_idxs: ", list1_idxs)
            # print("list2_idxs: ", list2_idxs)

            # remove NMS-suppressed detections
            #  also remove small-box detections
            area_threshold = float((W_orig*H_orig)/400.0)
            height_threshold = 20
            width_threshold = 20
            ratio_threshold = 0.8
            for idx in list2_idxs:
                cx, cy, width, height, area, as_ratio = box_metrics[idx]
                if width < width_threshold or height < height_threshold or area < area_threshold \
                    or as_ratio > ratio_threshold:
                    continue
                nms_scores.append(thresh_score[idx])
                nms_boxes.append(thresh_boxes[idx])
                nms_pred_cls.append(thresh_class[idx])
                self.car_boxes.append([cx, cy, width, height])
            #
        # end of if stmt to ensure at least one detection
        # return the final list of bounding boxes
        return self.car_boxes

if __name__ == '__main__':
        # Test the performance of the detector
        det =CarDetector()
        os.chdir(cwd)
        TEST_IMAGE_PATHS= glob(os.path.join('test_images/', '*.jpg'))

        for i, image_path in enumerate(TEST_IMAGE_PATHS[0:2]):
            print('')
            print('*************************************************')

            img_full = Image.open(image_path)
            img_full_np = det.load_image_into_numpy_array(img_full)
            img_full_np_copy = np.copy(img_full_np)
            start = time.time()
            b = det.get_localization(img_full_np, visual=False)
            end = time.time()
            print('Localization time: ', end-start)
#
