import numbers
import os
import numpy as np
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms.functional as VF
from albumentations.pytorch import ToTensorV2


class TripletFoodDataset(Dataset):

    def __init__(self, root, split='train', transforms=None, triplet=True, torchvision=True):
        self.root = root
        self.food_dir = os.path.join(root, 'food')
        self.imgs = os.listdir(self.food_dir)
        self.split = os.path.join(root, f'{split}_triplets.txt')
        self.samples = np.loadtxt(self.split, delimiter=' ').astype(int)
        self.triplet = triplet
        self.transforms = transforms
        self.torchvision = torchvision

    def __len__(self):
        if self.triplet:
            return len(self.samples)
        else:
            return len(self.imgs)

    def __getitem__(self, item):
        if not self.triplet:
            images = Image.open(os.path.join(self.food_dir, self.imgs[item]))
        else:
            image_paths = [os.path.join(self.food_dir, f'{i:05d}.jpg') for i in self.samples[item]]
            images = [Image.open(path) for path in image_paths]
        if self.transforms and self.torchvision:
            images = [self.transforms(image) for image in images]
        elif self.transforms and not self.torchvision:
            images = [np.array(image) for image in images]
            augments = [self.transforms(image=image) for image in images]
            images = [aug['image'] for aug in augments]
        return images


class MyPad(object):
    def __call__(self, image, target_size=(512, 512)):
        w, h = image.size
        tw, th = target_size
        wp = int((tw - w) / 2)
        hp = int((th - h) / 2)
        padding = [wp, hp, tw - w - wp, th - h - hp]
        return VF.pad(image, padding, 0, 'constant')


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        triplet_loss = distance_positive - distance_negative + self.margin
        num_pos_triplets = (triplet_loss > 0).float().sum()
        num_hard_triplets = (triplet_loss > self.margin).float().sum()
        num_semi_hard_triplets = num_pos_triplets - num_hard_triplets
        num_easy_triplets = (triplet_loss <= 0).float().sum()
        losses = F.relu(triplet_loss)
        return losses.mean(), [num_easy_triplets, num_semi_hard_triplets, num_hard_triplets]


class AverageNonzeroTripletsMetric:
    """
    Counts average number of nonzero triplets found in minibatches
    """

    def __init__(self):
        self.values = []

    def __call__(self, outputs, target, loss):
        self.values.append(loss)
        return self.value()

    def reset(self):
        self.values = []

    def value(self):
        return np.mean(np.asarray(self.values))

    def name(self):
        return 'Average nonzero triplets'
