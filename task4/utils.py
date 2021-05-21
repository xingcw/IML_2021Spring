import numbers
import os
import numpy as np
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms.functional as VF


class TripletFoodDataset(Dataset):

    def __init__(self, root, split='train', transforms=None, triplet=True):
        self.root = root
        self.food_dir = os.path.join(root, 'food')
        self.imgs = os.listdir(self.food_dir)
        self.split = os.path.join(root, f'{split}_triplets.txt')
        self.samples = np.loadtxt(self.split, delimiter=' ').astype(int)
        self.triplet = triplet
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        if not self.triplet:
            images = Image.open(os.path.join(self.food_dir, self.imgs[item]))
        else:
            image_paths = [os.path.join(self.food_dir, self.imgs[i]) for i in self.samples[item]]
            images = [Image.open(path) for path in image_paths]
        if self.transforms:
            images = [self.transforms(image) for image in images]
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

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class AverageNonzeroTripletsMetric:
    """
    Counts average number of nonzero triplets found in minibatches
    """

    def __init__(self):
        self.values = []

    def __call__(self, outputs, target, loss):
        self.values.append(loss.cpu().detach().numpy())
        return self.value()

    def reset(self):
        self.values = []

    def value(self):
        return np.mean(np.asarray(self.values))

    def name(self):
        return 'Average nonzero triplets'
