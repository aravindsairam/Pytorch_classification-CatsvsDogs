import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations
import numpy as np
from PIL import Image 
import os


class catsvsdogsTrain(Dataset):
    def __init__(self, folds, img_height, img_width, mean, std):
        super(catsvsdogsTrain, self).__init__()

        df = pd.read_csv("datasets/train_folds.csv")

        df = df[df.kfold.isin(folds)].reset_index(drop=True)

        self.files = df['files'].values

        self.labels = df['label'].values

        if len(folds) == 1:
            self.aug = albumentations.Compose([
                albumentations.Resize(img_height, img_width, always_apply = True),
                albumentations.HorizontalFlip(),
                albumentations.ShiftScaleRotate(rotate_limit = 30),
                albumentations.Normalize(mean, std, always_apply = True)
            ])
        else: 
            self.aug = albumentations.Compose([
                albumentations.Resize(img_height, img_width, always_apply = True),
                albumentations.Normalize(mean, std, always_apply = True)
            ])
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, item):
        images = Image.open(os.path.join("datasets/train", self.files[item]))
        target = self.labels[item]
        images =self.aug(image = np.array(images))['image']
        images = np.transpose(images, (2, 0, 1)).astype(np.float32)
        
        return {
            "images" : torch.tensor(images, dtype = torch.float),
            "targets" : torch.tensor(target),
        }
        
        
