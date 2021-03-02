import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from functools import partial
import matplotlib.pyplot as plt


class Salicon(nn.Module):
    def __init__(self):
        super(Salicon,self).__init__()
        self.Vgg16_fine_features=models.vgg16(pretrained=True).features[0:30]
        self.Vgg16_coarse_features=models.vgg16(pretrained=True).features[0:30]

        self.Unsample=partial(F.interpolate,mode='nearest')
        self.Sal_map=nn.Conv2d(1024,1,(1,1))
        self.Sigmoid=nn.Sigmoid()

    def forward(self,raw_img,resized_img):
        raw_features=self.Vgg16_fine_features(raw_img)
        resized_features=self.Vgg16_coarse_features(resized_img)
        H,W=raw_features.shape[-2],raw_features.shape[-1]

        resized_features=self.Unsample(resized_features,size=[H,W])

        out=torch.cat([raw_features,resized_features],dim=1)
        out=self.Sal_map(out)
        out=self.Sigmoid(out)

        return out


