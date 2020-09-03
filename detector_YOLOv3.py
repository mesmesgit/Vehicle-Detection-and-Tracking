# @updated by Michael Drolet 8/30/20
# @updated by Mark Strickland 9/1/20 to add YOLO
'''
Implement and test car detection (localization)
'''

import numpy as np
import cv2 as cv
import os
cwd = os.path.dirname(os.path.realpath(__file__))

model = cv.dnn.readNetFromDarknet('yolo_model/yolov3.cfg', 'yolo_model/yolov3.weights')
model.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
# determine the output layer
ln = model.getLayerNames()
ln = [ln[i[0] - 1] for i in model.getUnconnectedOutLayers()]
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
classes = COCO_INSTANCE_CATEGORY_NAMES
W_orig = 1280
H_orig = 720
# MES:  init list of classes of interest for traffic applications
list_coconame_traffic = ['person', 'bicycle', 'car', 'motorcycle',
                         'bus', 'truck']


class CarDetector(object):
    def __init__(self):
        #
        self.car_boxes = []
        self.W = W_orig
        self.H = H_orig
        os.chdir(cwd)

    # Helper function to convert image into numpy array
    def load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

    def get_localization(self, image):

        """Determines the locations of the cars in the image

        Args:
            image: camera image

        Returns:
            orig list of bounding boxes: coordinates [y_up, x_left, y_down, x_right]
            revised list of bounding boxes:  coord [cx, cy, width, height]

        """
        # open the input image
        img = image
        self.H, self.W = img.shape[:2]

        # initialize the returned list
        self.car_boxes = []
        nms_scores = []
        boxes = []
        box_confidences = []
        class_confidences = []
        score_threshold = float(0.4)    # orig: 0.5; 0.2 for sim/ss;  0.7 for vid
        nms_threshold = float(0.3)      # orig: 0.3; 0.7 for sim runs; 0.2 for ss runs; 0.2 for vid
        area_threshold = 0  # float((self.W*self.H)/400.0) = 2304
        height_threshold = 0  # 20
        width_threshold = 0  # 20
        ratio_threshold = 1.0  # 0.8
        n_topn = 10  # 15

        # FRCNN
        #  put the image into tensor form; start by creating a transformation
        # transform = T.Compose([T.ToTensor()])
        # MES - perform the transformation and send to GPU (if used)
        # img = transform(img).to(device)
        #  use the NN model to generate object detection prediction
        # pred = model([img])

        # YOLOv3
        # create blob input to network
        blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
        # set the blob as network input
        model.setInput(blob)
        # perform forward pass thru network
        outputs = model.forward(ln)

        # combine the 3 output groups into 1 (10647, 85)
        # large objects (507, 85)
        # medium objects (2028, 85)
        # small objects (8112, 85)
        outputs = np.vstack(outputs)

        # iterate thru each of rows of outputs, one bbox per row,
        #  and ignore detection below the confidence threshold
        for output in outputs:
            # output of network is: (centerx, centery, w, h), box conf, 80x(class conf)
            #
            # strip out the box confidence
            box_conf = output[4]
            # strip out only the class confidences
            scores = output[5:]
            # find the topn class confidences
            # n_topn = 15
            ind_topn = np.argpartition(scores, -n_topn)[-n_topn:]
            # iterate thru the indices of the top n
            for ind in ind_topn:
                classID = ind
                class_conf = scores[classID]
                #  also require that classID is traffic-related
                if (classes[classID] in list_coconame_traffic):
                    # consider box only if box confidence exceeds threshold
                    if box_conf > 0:
                        # if class confidence is below threshold, ignore
                        if class_conf > 0:
                            # scale outputs to size of original image
                            cx, cy, w, h = output[:4] * np.array([self.W, self.H, self.W, self.H])
                            # if very wide box at bottom of frame, skip
                            if w > 900 and cy > 500:
                                continue
                            # if very large box, skip
                            if w > 1000 and h > 550:
                                continue
                            if h > 700:
                                continue
                            # convert to x1, y1, x2, y2
                            x1 = int(cx - w//2)
                            y1 = int(cy - h//2)
                            x2 = int(x1 + w)
                            y2 = int(y1 + h)
                            # add to keep lists
                            boxes.append([x1, y1, x2, y2])
                            box_confidences.append(float(box_conf))
                            class_confidences.append(float(class_conf))
                        # end of if class_conf
                    # end of if box_conf
                # end of if classes[classID]
            # end of for ind in ind_topn
        # end loop for output
        # do the NMA suppression only if there is at least one detection:
        if len(boxes) > 0:
            # apply non maxima suppression to reduce duplicate boxes
            # since NMSBoxes requires boxes in [left, top, width, height] format, re-format the boxes
            boxes_rev = []
            scores_rev = []
            box_metrics = []
            for j, box in enumerate(boxes):
                (x1, y1, x2, y2) = box
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
                scores_rev.append(float(class_confidences[j]))
            #  make call to NMSBoxes
            idxs = cv.dnn.NMSBoxes(boxes_rev, scores_rev, score_threshold, nms_threshold)

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
            # area_threshold = float((W_orig*H_orig)/400.0)
            # height_threshold = 20
            # width_threshold = 20
            # ratio_threshold = 0.8
            for idx in list2_idxs:
                cx, cy, width, height, area, as_ratio = box_metrics[idx]
                if width < width_threshold or height < height_threshold or area < area_threshold \
                 or as_ratio > ratio_threshold:
                    continue
                nms_scores.append(class_confidences[idx])
                self.car_boxes.append([cx, cy, width, height])
            #
        # end of if stmt to ensure at least one detection

        # return the final list of bounding boxes
        return self.car_boxes, nms_scores
    # end get_localization()
# end class CarDetector
