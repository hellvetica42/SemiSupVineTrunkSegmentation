import os
from typing import Union

import numpy as np
import torchvision
import torchvision.transforms as tr
from PIL import Image

from dataset.transforms import Normalize

from .common import train_cities, city_classes


def string_to_sequence(s: str, dtype=np.int32) -> np.ndarray:
    return np.array([ord(c) for c in s], dtype=dtype)


def sequence_to_string(seq: np.ndarray) -> str:
    return "".join([chr(c) for c in seq])


def pack_sequences(seqs: Union[np.ndarray, list]):
    values = np.concatenate(seqs, axis=0)
    offsets = np.cumsum([len(s) for s in seqs])
    return values, offsets


def unpack_sequence(values: np.ndarray, offsets: np.ndarray, index: int) -> np.ndarray:
    off1 = offsets[index]
    if index > 0:
        off0 = offsets[index - 1]
    elif index == 0:
        off0 = 0
    else:
        raise ValueError(index)
    return values[off0:off1]


class dataloader(torchvision.datasets.VisionDataset):
    def __init__(
        self,
        root,
        image_set,
        transforms=None,
        transform=None,
        target_transform=None,
        label_state=True,
        data_set="vine",
        mask_type=".png",
    ):
        super().__init__(root, transforms, transform, target_transform)
        self.mask_type = mask_type
        self.data_set = data_set
        self.has_label = label_state

        if data_set == 'vine':
            self._vine_init(root, image_set)
        else:
            raise ValueError

        self.ignore_index = 255

        # Different label states
        self.color_jitter = tr.ColorJitter(0.1, 0.1, 0.1, 0.05)
        self.to_gray = tr.RandomGrayscale(p=0.2)

    def encode_target(self, target):
        return self.id_to_train_id[np.array(target)].astype(np.uint8)

    def get_image_and_target(self, image_path, mask_path):
        img = Image.open(image_path).convert("RGB")
        if mask_path is None:
            target = np.zeros((128,128,1))
        else:
            target = Image.open(mask_path)

        return img, target

    def __getitem__(self, index):

        if self.has_label:
            #print(self.image_paths[index], self.mask_paths[index])
            img, target = self.get_image_and_target(self.image_paths[index], self.mask_paths[index])
            img, target = self.transforms(img, target)
            return img, target
        else:
            img, target = self.get_image_and_target(self.image_paths[index], None)
            img, target = self.transforms(img, target)
            img_aug = self.to_gray(self.color_jitter(img))
            img, target = Normalize()(img, target)
            img_aug = Normalize()(img_aug)
            return img, target, img_aug

    def __len__(self):
        return self.image_len

    def _vine_init(self, root, image_set):
        # It's tricky, lots of folders
        if self.has_label:
            if image_set == 'train':
                image_split = os.path.join(root, 'labeled-train-small.txt')

            if image_set == 'valid':
                image_split = os.path.join(root, 'labeled-valid-small.txt')

            print(image_set)
            
            with open(image_split, 'r') as f:
                image_paths = [os.path.join(root, x.strip()) for x in f.readlines()]
                mask_paths = [i[:-4]+"_mask.png" for i in image_paths]
                self.mask_paths = sorted(mask_paths)
        else:
            image_split = os.path.join(root, 'unlabeled-large.txt')
            self.mask_paths = None

            with open(image_split, 'r') as f:
                image_paths = [os.path.join(root, x.strip()) for x in f.readlines()]

        self.image_paths = sorted(image_paths)
        self.image_len = len(self.image_paths)
        print(image_set, self.image_len)


    def _city_init(self, root, image_set):
        # It's tricky, lots of folders
        image_dir = os.path.join(root, "leftImg8bit")
        check_dirs = False
        mask_dir = os.path.join(root, "gtFine")

        if image_set == "val" or image_set == "test":
            image_dir = os.path.join(image_dir, image_set)
            mask_dir = os.path.join(mask_dir, image_set)
        else:
            image_dir = os.path.join(image_dir, "train")
            mask_dir = os.path.join(mask_dir, "train")

        if check_dirs:
            for city in train_cities:
                temp = os.path.join(mask_dir, city)
                if not os.path.exists(temp):
                    os.makedirs(temp)

        # We first generate data lists before all this, so we can do this easier
        splits_dir = os.path.join(root, "data_lists")
        split_f = os.path.join(splits_dir, image_set + ".txt")
        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        print(split_f)
        ################# 100% #################
        images = [os.path.join(image_dir, x + "_leftImg8bit.png") for x in file_names]
        print("labeled 100% : ", len(images))
        self.image_len = len(images)
        img_seq = [string_to_sequence(s) for s in images]
        self.image_v, self.image_o = pack_sequences(img_seq)

        masks = [
            os.path.join(mask_dir, x + "_gtFine_labelIds" + self.mask_type)
            for x in file_names
        ]
        mask_seq = [string_to_sequence(s) for s in masks]
        self.masks_v, self.masks_o = pack_sequences(mask_seq)
        #########################################
