from tqdm import tqdm
import torch
import cv2
import numpy as np

import segmentation_models_pytorch as smp
from TorchNode import TorchDetector


class VineSegmentation:

    def __init__(self) -> None:
        #ENCODER = 'mobilenet_v2'
        ENCODER = 'efficientnet-b0'
        ENCODER_WEIGHTS = 'imagenet'
        CLASSES = ['trunk']
        ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
        self.DEVICE = 'cuda'

        # create segmentation model with pretrained encoder
        self.model = torch.load('best_model.pth').to(self.DEVICE)
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

        self.detector = TorchDetector()

        self.resize_factor = 2

    def to_tensor(self, x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')

    def detect(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (frame.shape[1]//self.resize_factor, frame.shape[0]//self.resize_factor)).astype('uint8')

        crops, crop_datas = self.get_crops(frame)
        if len(crops) == 0:
            return frame
        masks = self.get_masks(crops, crop_datas)
        seg_frame = self.merge_segmentations(frame, masks, crop_datas)
        return seg_frame

    def get_crops(self, frame):

        crops = []
        crop_datas = []

        # Get bounding boxes
        bboxes = self.detector.detect(frame)
        # Get crops of original frame from bounding boxes
        for bbox in bboxes:
            pxcenter = (bbox.xmin + bbox.xmax)//2
            pycenter = (bbox.ymin + bbox.ymax)//2
            pwidth = bbox.xmax - bbox.xmin
            pheight = bbox.ymax - bbox.ymin

            orig_data = {'width':pwidth, 'height':pheight, 'bbox':bbox}

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
            
            square_shape = crop.shape[:2]
            orig_data['square'] = square_shape
            crop_datas.append(orig_data)

            #resize to model input and format
            crop = cv2.resize(crop, (128,128))
            x_tensor = self.preprocessing_fn(crop)
            x_tensor = self.to_tensor(x_tensor)
            crops.append(x_tensor)

        crops = np.array(crops)
        return crops, crop_datas

    def get_masks(self, crops, crop_datas):
        x_tensor = torch.from_numpy(crops).to(self.DEVICE)
        pr_masks = self.model.predict(x_tensor)
        pr_masks = (pr_masks.squeeze().cpu().numpy().round())

        masks = []
        num_masks = pr_masks.shape[0] if len(pr_masks.shape) > 2 else 1
        for i in range(num_masks):
            tmp = cv2.resize(pr_masks[i], crop_datas[i]['square'])

            halfpoint = crop_datas[i]['square'][0]//2
            tmp = tmp[
                halfpoint - crop_datas[i]['height']//2 : halfpoint + crop_datas[i]['height']//2,
                halfpoint - crop_datas[i]['width'] //2 : halfpoint + crop_datas[i]['width'] //2,
            ]
            masks.append(tmp)

        return masks

    def merge_segmentations(self, frame, masks, crop_datas):
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for i, m in enumerate(masks):
            tmp = cv2.cvtColor(m, cv2.COLOR_GRAY2RGB)
            tmp = cv2.resize(tmp, (crop_datas[i]['width'], crop_datas[i]['height'])) * 255
            tmp = tmp.astype('uint8')
            contours, hierarchy = cv2.findContours(tmp[:, :, 0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            crop = frame[
                crop_datas[i]['bbox'].ymin : crop_datas[i]['bbox'].ymax,
                crop_datas[i]['bbox'].xmin : crop_datas[i]['bbox'].xmax
            ]
            #dst = cv2.addWeighted(crop, 0.5, tmp, 0.5, 0.0)
            dst = cv2.drawContours(crop, contours, -1, (255, 0, 0), 2)
            frame[
                crop_datas[i]['bbox'].ymin : crop_datas[i]['bbox'].ymax,
                crop_datas[i]['bbox'].xmin : crop_datas[i]['bbox'].xmax
            ] = dst
        return frame

if __name__ == "__main__":
    vineseg = VineSegmentation()

    #video_path = 'istra1.mp4'
    video_path = '../../dalmacija/VID_20220414_181709.mp4'
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    print(frame_width, frame_height)

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('outputtesst.mp4',fourcc, 30.0, (frame_width,frame_height))


    for _ in tqdm(range(length)):
        ret, frame = cap.read()
        frame=cv2.resize(frame, (3840, 2160))
        if ret != True:
            break

        seg_frame = vineseg.detect(frame)
        seg_frame = cv2.resize(seg_frame, (frame_width, frame_height))
        seg_frame = cv2.cvtColor(seg_frame, cv2.COLOR_RGB2BGR)
        out.write(seg_frame)

        cv2.imshow("FRAME", seg_frame)
        char = cv2.waitKey(1)
        if char == ord('q'):
            break
        elif char == ord('n'):
            for _ in range(60):
                ret, frame = cap.read()
                

    out.release()
    cap.release()
    cv2.destroyAllWindows()