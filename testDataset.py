# -*- coding:utf-8 -*-
from torch.utils import data
from torchvision import transforms as T
import torch
from PIL import Image
import numpy as np
import os
import cv2

# 数据集，
class ReadData(data.Dataset):
    def __init__(self,transforms=None,file_list=None,labels = None):
        self.images_path = file_list  # 图片路径a
        self.images_labels = labels  # 图片的标签
        # self._read_file() # 读取数据
        if transforms is None:
            normalize = T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            self.transforms = T.Compose([
                T.Resize((32, 32)),
                T.ToTensor(),
                normalize
            ])

    def _read_file(self):
        # for file in self.images_path:
        pass
        # 设置路径和标签

    def __getitem__(self, index):
        img_path = self.images_path[index]
        label = self.images_labels[index]

        img = Image.open(img_path)
        img = img.convert('RGB')
        data = self.transforms(img) # 转换

        return data, int(label) # 返回数据.标签

    def __len__(self):
        return len(self.images_path)
