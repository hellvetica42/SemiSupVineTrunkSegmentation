import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import os
import time
from tqdm import tqdm

def to_tensor(x):
    return x.transpose(2, 0, 1).astype('float32')

def eval_performance(ENCODER, ARCH):
    ENCODER_WEIGHTS = "imagenet"
    DEVICE = 'cuda'

    MODEL_PATH = f"SEG_MODELS/best_model_{ARCH}_{ENCODER}.pth"

    model = torch.load(MODEL_PATH).to(DEVICE)
    model.eval()
    preprocessing = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    DATA_PATH = '../ExtractedData/dalmacija-large-128'
    filepaths = os.listdir(DATA_PATH)[:100]
    np_images = []
    for f in tqdm(filepaths):
        img = cv2.imread(os.path.join(DATA_PATH, f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x_tensor = preprocessing(img)
        x_tensor = to_tensor(x_tensor)
        np_images.append(x_tensor)

    np_images = np.array(np_images)
    x_tensor = torch.from_numpy(np_images).to(DEVICE)
    times = []
    for i in range(10):
        start_time = time.perf_counter()
        model.predict(x_tensor)
        end_time = time.perf_counter()
        times.append(end_time-start_time)
    print(ARCH, ENCODER, sum(times)/len(times))


if __name__ == "__main__":
    ENCODERS = [
        'efficientnet-b0',
        'timm-mobilenetv3_large_075'
    ]
    ARCHS = [
        'DeepLabV3Plus',
        'Link',
        'MANET',
        'PAN',
        'PSP',
        'UnetPlusPlus'
    ]

    for arch in ARCHS:
        for ENC in ENCODERS:
            eval_performance(ENC, arch)


