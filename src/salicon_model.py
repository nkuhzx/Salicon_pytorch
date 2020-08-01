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

# from torchviz import make_dot
# if __name__ == '__main__':
#
#     use_cuda='cuda' if torch.cuda.is_available() else 'cpu'
#
#     fake_image_corse=torch.randn(300,400,3)
#     fake_image_corse = fake_image_corse.permute(2, 0, 1).unsqueeze(0).to(use_cuda)
#
#     fake_image=torch.randn(600,800,3)
#     fake_image=fake_image.permute(2,0,1).unsqueeze(0).to(use_cuda)
#
#
#     # print(fake_image.shape)
#     net=Salicon()
#     net.to(use_cuda)
#     #saliency_image=net.forward(fake_image,fake_image_corse)
#     #saliency_image=saliency_image.squeeze()
#     #saliency_image=saliency_image.cpu().detach().numpy()
#
#     dummy_fine=torch.rand(1,3,600,800).to(use_cuda)
#     dummy_coarse=torch.rand(1,3,300,400).to(use_cuda)
#     dummy_output=net(dummy_fine,dummy_coarse)
#     g=make_dot(dummy_output)
#     g.render('ourmodel',view=False)
#     #plt.imshow(saliency_image,cmap='gray')
#     #plt.show()
#     #net.forward(fake_image_corse)

