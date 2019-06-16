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
    def __init__(self,transforms=None, train=True):
        self.train = train
        self.images_path = []  # 图片路径
        self.images_labels = []  # 图片的标签
        self._read_file() # 读取数据
        if transforms is None:
            normalize = T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )

            if not train:  # 测试集的话
                self.transforms = T.Compose([
                    T.Resize((32, 32)),
                    T.ToTensor(),
                    normalize
                ])
            else:  # 训练集
                self.transforms = T.Compose([
                    T.Resize((32, 32)),
                    # T.Scale(224),
                    # T.RandomSizedCrop(224),
                    # T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])

    def _read_file(self):
        data_path = './data/'
        if self.train:
            # 获得训练集,并将图片路径放入变量中
            # store = Store()
            # self.images_path = store.get_train_images_path()
            # self.images_labels = store.get_train_labels()
            files = os.listdir(data_path)
            for file in files:
                file_path = data_path + file
                self.images_path.append(file_path)
                label = int(file.strip().split("_")[0])  # 获得标签
                label -= 1  # label -1 标签从0开始
                self.images_labels.append(label)
        else:  # 测试集和训练集一样
            data_path = "./testdata/" # 测试文件的路径
            print("获得测试文件")
            files = os.listdir(data_path)
            for file in files:
                file_path = data_path + file
                self.images_path.append(file_path)
                label = int(file.strip().split("_")[0])  # 获得标签
                label -= 1  # label -1 标签从0开始
                self.images_labels.append(label)
            # pass # 测试，将测试的图片和label放入变量中
        # 设置路径和标签

    def __getitem__(self, index):
        img_path = self.images_path[index]
        label = self.images_labels[index]
        # img = cv2.imread(img_path)
        # img = cv2.resize(img, (28, 28))
        img = Image.open(img_path)
        img = img.convert('RGB')
        # img = img/255.
        # data = img[:, :, :0]  # 通道一样,随便选一个
        # data = torch.from_numpy(data)
        data = self.transforms(img) # 转换
        return data, int(label) # 返回数据.标签

    def __len__(self):
        return len(self.images_path)
