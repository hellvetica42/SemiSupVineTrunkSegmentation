#!/usr/bin/python3
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# Python imports
import numpy as np
import cv2

# ROS imports
import scipy.io as sio
# Deep Learning imports
import torch
import yaml
from numpy import random
from skimage.transform import resize
from torch.autograd import Variable

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import (apply_classifier, check_img_size,
                           check_requirements, increment_path,
                           non_max_suppression, scale_coords, set_logging,
                           strip_optimizer)
from utils.plots import plot_one_box
# util + model imports
from utils.torch_utils import load_classifier, select_device, time_synchronized
import tensorflow as tf

class BoundingBox:
    def __init__(self) -> None:
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None     
        self.Class = None
        self.probability = None
class TorchDetector:

    def __init__(self):
        print("Node yolov5 started")

        # Load weights parameter
        self.weights_path = 'best.pt'

        # Raise error if it cannot find the model
        if not os.path.isfile(self.weights_path):
            raise IOError(('{:s} not found.').format(self.weights_path))

        # Load image parameter and confidence threshold


        self.conf_thres = 0.5

        # Load other parameters
        self.device_name = 'cuda:0'
        self.device = select_device(self.device_name)
        self.gpu_id = 0
        self.network_img_size = 416
        self.iou_thres = 0.45
        self.augment = True

        self.classes = None
        self.agnostic_nms = False

        self.w = 0
        self.h = 0

        self.half = self.device.type != 'cpu'


        self.model = attempt_load(self.weights_path, map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.network_img_size = check_img_size(self.network_img_size, s=self.stride)

        self.model = attempt_load(self.weights_path, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())      # model stride
        self.network_img_size = check_img_size(self.network_img_size, s=self.stride)  # check img_size

        if self.half:
            self.model.half()

        self.names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255)
                        for _ in range(3)] for _ in self.names]

                                        
        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.network_img_size, self.network_img_size).to(
                self.device).type_as(next(self.model.parameters())))  # run once


    def detect(self, img):
        
        input_img = self.preprocess(img)
        input_img = Variable(input_img.type(torch.HalfTensor))
        input_img = input_img.to(self.device)

        with torch.no_grad():
            detections= self.model(input_img)[0]
            detections = non_max_suppression(detections, self.conf_thres, self.iou_thres,
                                            classes=self.classes, agnostic=self.agnostic_nms)

        detection_results = []
        # Parse detections
        if detections[0] is not None:
            for detection in detections[0]:
                # Get xmin, ymin, xmax, ymax, confidence and class
                xmin, ymin, xmax, ymax, conf, det_class = detection
                pad_x = max(self.h - self.w, 0) * \
                    (self.network_img_size/max(self.h, self.w))
                pad_y = max(self.w - self.h, 0) * \
                    (self.network_img_size/max(self.h, self.w))
                unpad_h = self.network_img_size-pad_y
                unpad_w = self.network_img_size-pad_x
                xmin_unpad = ((xmin-pad_x//2)/unpad_w)*self.w
                xmax_unpad = ((xmax-xmin)/unpad_w)*self.w + xmin_unpad
                ymin_unpad = ((ymin-pad_y//2)/unpad_h)*self.h
                ymax_unpad = ((ymax-ymin)/unpad_h)*self.h + ymin_unpad

                # Populate darknet message
                detection_msg = BoundingBox()
                detection_msg.xmin = int(xmin_unpad)
                detection_msg.xmax = int(xmax_unpad)
                detection_msg.ymin = int(ymin_unpad)
                detection_msg.ymax = int(ymax_unpad)
                detection_msg.probability = float(conf)
                detection_msg.Class = self.names[int(det_class)]

                # Append in overall detection message
                detection_results.append(detection_msg)

                # Publish detection results

        return detection_results


    def preprocess(self, img):
    # Extract image and shape
        img = np.copy(img)
        img = img.astype(float)
        height, width, channels = img.shape
        if (height != self.h) or (width != self.w):
            self.h = height
            self.w = width
            # Determine image to be used
            self.padded_image = np.zeros(
                (max(self.h, self.w), max(self.h, self.w), channels)).astype(float)

        # Add padding
        if (self.w > self.h):
            self.padded_image[(self.w-self.h)//2: self.h +
                            (self.w-self.h)//2, :, :] = img
        else:
            self.padded_image[:, (self.h-self.w)//2: self.w +
                            (self.h-self.w)//2, :] = img
        # Resize and normalize
        input_img = resize(self.padded_image, (self.network_img_size, self.network_img_size, 3))/255.

        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))

        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()
        input_img = input_img[None]

        return input_img

    def visualize(self, output, imgIn):
        # Copy image and visualize
        imgOut = imgIn.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.8
        thickness = 2
        for index in range(len(output)):
            label = output[index].Class
            x_p1 = output[index].xmin
            y_p1 = output[index].ymin
            x_p3 = output[index].xmax
            y_p3 = output[index].ymax
            confidence = output[index].probability

            # Set class color
            color = self.colors[self.names.index(label)]



            # Create rectangle
            cv2.rectangle(imgOut, (int(x_p1), int(y_p1)), (int(x_p3), int(
                y_p3)), (color[0], color[1], color[2]), thickness)
            text = ('{:s}: {:.3f}').format(label, confidence)
            cv2.putText(imgOut, text, (int(x_p1), int(y_p1+20)), font,
                        fontScale, (255, 255, 255), thickness, cv2.LINE_AA)

        return imgOut

if __name__ == '__main__':
    detector = TorchDetector()

    #videoName = 'dalmacija2.mp4'
    #videoPath = os.path.join('..', 'Videos', videoName)
    videoName = 'istra2.mp4'
    videoPath = videoName

    cap = cv2.VideoCapture(videoPath)

    frame_count = 0
    while True:
        frame_count += 1
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if ret != True:
            break

        outimg, bboxes = detector.detect(frame)
        bboxcount = 0
        for bbox in bboxes:
            continue
            bboxcount += 1
            pxcenter = (bbox.xmin + bbox.xmax)//2
            pycenter = (bbox.ymin + bbox.ymax)//2
            pwidth = bbox.xmax - bbox.xmin
            pheight = bbox.xmax - bbox.xmin

            if pwidth > pheight:
                pheight = pwidth
            elif pheight > pwidth:
                pwidth = pheight

            if pycenter - pheight//2 < 0 or pycenter + pheight//2 > frame.shape[0] or \
               pxcenter - pwidth//2 < 0  or pxcenter + pwidth//2 > frame.shape[1]:
               continue

            crop = frame[
                            pycenter - pheight//2 : pycenter + pheight//2,
                            pxcenter - pwidth//2  : pxcenter + pwidth//2 
                        ]
            crop = cv2.resize(crop, (128,128))
            crop = np.array(crop)

        outimg = cv2.cvtColor(outimg, cv2.COLOR_RGB2BGR)
        try:
            #crop = tf.image.resize_with_pad(crop, 128, 128)
            cv2.imshow("IMG", outimg)
            char = cv2.waitKey(1)
        except Exception as e:
            print(e)
            exit()

        if char == ord('q'):
            break
        elif char == ord('n'):
            for i in range(60):
                ret, frame = cap.read()
        elif char == ord('s'):
            cv2.imwrite(f"IstraDetections/frame_{frame_count}.png", outimg)