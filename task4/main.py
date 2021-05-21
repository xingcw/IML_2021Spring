import torch
from tqdm import tqdm
import pandas as pd
from torchvision.transforms import Pad, Normalize, Resize, Compose, ToTensor, RandomHorizontalFlip, \
    RandomErasing, RandomRotation, CenterCrop
from torch.utils.data import DataLoader, random_split
from networks import EmbeddingNet, TripletNet
from trainer import fit
from utils import *


def train_local():
    torch.manual_seed(123)
    # ================ get all images statistics ===============
    """
    Trainset = TripletFoodDataset('dataset', 'train', triplet=False)
    Testset = TripletFoodDataset('dataset', 'test', triplet=False)
    max_w, max_h = max([im.size[0] for im in Trainset]), max([im.size[1] for im in Trainset])
    MEAN = np.mean(np.asarray([np.mean(np.asarray(im).reshape(-1, 3), 0) for im in Trainset]), axis=0)
    STD = np.std(np.asarray([np.mean(np.asarray(im).reshape(-1, 3), 0) for im in Trainset]), axis=0)
    print(MEAN, STD)
    """
    # max size: [512, 342], mean: [155.08, 131.61, 105.13], std: [32.44, 31.13, 33.46]
    # ======================== preprocessing ====================
    transforms = Compose([MyPad(), Resize((224, 224)), ToTensor(),
                          Normalize(mean=(155.08, 131.61, 105.13), std=(32.44, 31.13, 33.46))])
    Trainset = TripletFoodDataset('dataset', 'train', transforms)
    train_size = int(0.8 * len(Trainset))
    val_size = len(Trainset) - train_size
    train_set, val_set = random_split(Trainset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=200, shuffle=True)

    # ==================== get model and train ==================
    backbone = 'resnet50'
    emdnet = EmbeddingNet(backbone, pretrain=False)
    model = TripletNet(emdnet).cuda()
    print(model)
    adam = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(adam, patience=20, threshold=0.005)
    metric = AverageNonzeroTripletsMetric()
    loss_fn = TripletLoss(0.3)
    fit(train_loader, val_loader, model, loss_fn, adam, scheduler, 100, True, 1, backbone, metrics=[metric])


def train_remote():
    # Get scratch dir where to store files on euler
    tmpdir = os.getenv('TMPDIR')
    os.system(f'tar -xvf dataset.tar -C {tmpdir}')
    torch.manual_seed(123)
    transforms = Compose([MyPad(), Resize((256, 256)), CenterCrop(224), RandomHorizontalFlip(0.2), RandomRotation(90),
                          ToTensor(), Normalize(mean=(155.08, 131.61, 105.13), std=(32.44, 31.13, 33.46))])
    # RandomErasing()])
    # Running on ETH euler
    Trainset = TripletFoodDataset(tmpdir + '/dataset', 'train', transforms)
    train_size = int(0.8 * len(Trainset))
    val_size = len(Trainset) - train_size
    train_set, val_set = random_split(Trainset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=500, shuffle=True)

    # ==================== get model and train ==================
    backbone = 'resnet50'
    emdnet = EmbeddingNet(backbone)
    model = TripletNet(emdnet).cuda()
    print(model)
    adam = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(adam, patience=20, threshold=0.005)
    metric = AverageNonzeroTripletsMetric()
    loss_fn = TripletLoss(0.3)
    fit(train_loader, val_loader, model, loss_fn, adam, scheduler, 100, True, 2, backbone,
        val_thresh=0.25, metrics=[metric])


def predict():
    transforms = Compose([MyPad(), Resize((224, 224)), ToTensor(),
                          Normalize(mean=(155.08, 131.61, 105.13), std=(32.44, 31.13, 33.46))])
    Testset = TripletFoodDataset('dataset', 'test', transforms)
    test_loader = DataLoader(Testset, batch_size=200, shuffle=False)
    saved_model = torch.load('models/best_resnet50.pth')
    backbone = 'resnet50'
    emdnet = EmbeddingNet(backbone)
    model = TripletNet(emdnet).cuda()
    model.load_state_dict(saved_model['model_state_dict'], strict=False)
    model.eval()
    preds = []
    for images in tqdm(test_loader, total=len(test_loader)):
        data = tuple(d.cuda() for d in images)
        emd1, emd2, emd3 = model(*data)
        dist12 = (emd1 - emd2).pow(2).sum(1)
        dist13 = (emd1 - emd3).pow(2).sum(1)
        pred = torch.le(dist12, dist13).float().cpu().numpy().tolist()
        preds.extend(pred)
    np.savetxt('pred_1.txt', np.asarray(preds), fmt='%d')


if __name__ == '__main__':
    # train_local()
    # train_remote()
    predict()
