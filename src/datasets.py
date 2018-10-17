# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 13:40:34 2018

@author: USER
"""

import os
import cv2
from utils import *
import pickle

import random
import torch
import torchvision.transforms as Transform
from torchvision.utils import save_image

class Anime:
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.img_files = os.listdir(self.root_dir)
        self.dataset_len = len(self.img_files)
        
        self.transform = transform
        
    def length(self):
        return self.dataset_len
    
    def get_item(self, idx):
        img_path = os.path.join(self.root_dir, self.img_files[idx])
        img = cv2.imread(img_path)
        img = img[:, :, (2, 1, 0)]
        if self.transform:
            img = self.transform(img)
        return img

class Shuffler:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataset_len = self.dataset.length()
    
    def get_batch(self):
        indices = random.sample(range(self.dataset_len), self.batch_size)
        batch = [self.dataset.get_item(i).unsqueeze(0) for i in indices]
        batch = torch.cat(batch, 0)
        return batch
    
if __name__ == '__main__':
    
    batch_size = 5
    transform = Transform.Compose([Transform.ToTensor(),
                                   Transform.Normalize((0.5, 0.5, 0.5),
                                                       (0.5, 0.5, 0.5))])
    data = Anime('../cropped', transform)
    
    shuffler = Shuffler(data, batch_size)
    img = shuffler.get_batch()
    save_image(denorm(img), './test.png')
