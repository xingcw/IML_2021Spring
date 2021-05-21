import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class EmbeddingNet(nn.Module):
    def __init__(self, model=None, pretrain=True):
        super(EmbeddingNet, self).__init__()
        if model == 'resnet18':
            self.convnet = models.resnet18(pretrained=pretrain)
        elif model == 'inception_v3':
            self.convnet = models.inception_v3(pretrained=pretrain)
        elif model == 'vgg16':
            self.convnet = models.vgg16(pretrained=pretrain)
        elif model == 'mobilenet_v2':
            self.convnet = models.mobilenet_v2(pretrained=pretrain)
        else:
            self.convnet = nn.Sequential(nn.Conv2d(3, 32, 7), nn.PReLU(),
                                         nn.MaxPool2d(2, stride=2),
                                         nn.Conv2d(32, 32, 5), nn.PReLU(),
                                         nn.MaxPool2d(2, stride=2),
                                         nn.Conv2d(32, 32, 3), nn.PReLU(),
                                         nn.MaxPool2d(2, stride=2)
                                         )
        self.fc = nn.Sequential(nn.Linear(1000, 64),
                                nn.PReLU(),
                                nn.Dropout(0.3),
                                nn.Linear(64, 64))

    def forward(self, x):
        output = self.convnet(x)
        # output = output.view(output.size()[0], -1)
        # output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)
