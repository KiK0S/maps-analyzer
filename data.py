import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from torch.utils.data import Dataset, DataLoader
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
import random
import json
import math
from tqdm import tqdm
from torchvision.io import read_image
from sklearn.cluster import DBSCAN, KMeans
import numpy as np
import torch.optim as optim
import pickle
from IPython.display import clear_output
import albumentations as A
import segmentation_models_pytorch as smp
from datetime import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def classnum(shape):
    if shape['label'] == 'fence':
        return 1
    if shape['label'] == 'forbidden':
        return 1
    if shape['label'] == 'building':
        return 1
    if shape['label'] == 'water':
        return 1
    return 1

def vec(x0, y0, x1, y1):
    return x0 * y1 - x1 * y0

def calc_vec(a, b, c):
    return vec(b[0] - a[0], b[1] - a[1], c[0] - a[0], c[1] - a[1])

def intersec(a, b, c, d):
    return calc_vec(a, b, c) * calc_vec(a, b, d) < 0 and calc_vec(c, d, b) * calc_vec(c, d, a) < 0

def prepare(pic):
    res = pic / 255
    return res

def detect(x, y, shape):
    if shape['shape_type'] == 'Line':
        v0, v1 = shape['points']
        a = v1[1] - v0[1]
        b = v0[0] - v1[0]
        c = -(v1[0] * a + v1[1] * b)
        if a ** 2 + b ** 2 == 0:
            return False
        d = (a * x + b * y + c) / math.sqrt(a ** 2 + b ** 2)
        between = ((x - v0[0]) * (v1[0] - v0[0]) + (y - v0[1]) * (v1[1] - v0[1]) ) >= 0 and ((x - v1[0]) * (v0[0] - v1[0]) + (y - v1[1]) * (v0[1] - v1[1]) ) >= 0
        return abs(d) < 3 and between
    if shape['shape_type'].startswith('line'):
        for i in range(len(shape['points']) - 1):
            line = {'points': [shape['points'][i], shape['points'][i + 1]],
                    'shape_type': 'Line'}
            if detect(x, y, line):
                return True
        return False
    if shape['shape_type'] == 'polygon':
        s = 0
        other = [10002.7, 10000.5] 
        for i in range(len(shape['points'])):
            point = shape['points'][i]
            next_point = shape['points'][(i + 1) % len(shape['points'])]
            if intersec([x, y], other, point, next_point):
                s += 1
        if s % 2 != 0:
            return True
        return False
    return False

class DraftDataset(Dataset):
    def __init__(self, annotations_file, data_dir, transform=None, target_transform=None):
        self.labels = pd.read_csv(annotations_file)
        self.img_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.length = self.labels.shape[0]
        self.size = 100
        self.images = []
        self.annotations = []
        if os.path.isfile('data/X.data') and os.path.isfile('data/y.data'):
            self.X = pickle.load(open('data/X.data', 'rb'))
            self.y = pickle.load(open('data/y.data', 'rb'))
            return
        self.masks = []
        self.masks_processed = []
        for idx in range(self.length):
            assert idx < len(self.labels)
            img_path = os.path.join(self.img_dir, self.labels.iloc[idx, 0] + '.jpg')
            print('Processing {0}'.format(img_path))
            pic = cv2.imread(img_path)
            b, g, r = cv2.split(pic)
            self.images.append(cv2.merge([r, g, b]))
        for idx in range(self.length):
            assert idx < len(self.labels)
            img_path = os.path.join(self.img_dir, self.labels.iloc[idx, 0] + '.jpg')
            print('Processing {0}'.format(img_path))
            pic = cv2.imread(img_path)
            b, g, r = cv2.split(pic) # по умолчанию cv2 почему-то отдает цвета в порядке BGR вместо RGB
            self.images.append(cv2.merge([r, g, b]))
            self.annotations.append(json.load(open(data_dir + self.labels.iloc[idx, 0] + '.json', 'r')))
            img = self.images[-1]
            annotation = self.annotations[-1]
            img_shape = img.shape
            mask = np.zeros(shape=(img_shape[0], img_shape[1]))
            for i in tqdm(range(img_shape[0]), desc='X loop', position=0, leave=True):
                for j in range(img_shape[1]):
                    for shape in annotation['shapes']:
                        if detect(j, i, shape): # проблема с индексацией labelme
                            mask[i][j] = float(classnum(shape))
                            break
            self.masks.append(mask)
            print('Saving')
            pickle.dump(self.masks, open('data/y.data', 'wb'))
            clear_output(wait=True)
        self.X = self.images
        self.y = self.masks
        pickle.dump(dataset.y, open('data/Y.data', 'wb'))
        pickle.dump(dataset.X, open('data/X.data', 'wb'))
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = idx % self.length
        img = self.X[idx]
        mask = self.y[idx]
        return img, mask




class AugmentedDataset(Dataset):
    def __init__(self, draft, augmentor, size = 250):
        self.draft = draft
        self.augmentor = augmentor
        self.size = size
    
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        X, y = self.draft[idx]
        augmented = self.augmentor(image=X, mask=y)
        X = (torch.tensor(augmented['image']).permute(2, 0, 1) / 256).float()
        y = torch.tensor(augmented['mask']).float().view(1, 256, 256)
        return X, y

def get_data():
    dataset = DraftDataset('data/annotations.csv', 'data/')

    aug = A.Compose([
        A.RandomCrop(width=256, height=256, p=1),
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.GaussNoise(p=.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=.1),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.RandomBrightnessContrast(),            
        ], p=0.3),  
    ])
    data = AugmentedDataset(dataset, aug)
    return data
