import numpy as np
import torch
import torch.nn as nn
import cv2

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"..")))

import argparse

from src.data_utils import getTest_loader
from src.salicon_model import Salicon



def test(model,device,test_loader):
    model.eval()

    with torch.no_grad():
        for input,target in test_loader:
            fine_img,coarse_img=input
            fine_img=fine_img.unsqueeze(0).to(device)
            coarse_img=coarse_img.unsqueeze(0).to(device)

            pred=model(fine_img,coarse_img)

            pred_image=pred.squeeze()
            h,w=target.shape[-2],target.shape[-1]

            smap=(pred_image-torch.min(pred_image))/((torch.max(pred_image)-torch.min(pred_image)))
            smap=smap.cpu().numpy()
            smap=cv2.resize(smap,(w,h),interpolation=cv2.INTER_CUBIC)
            smap=cv2.GaussianBlur(smap,(75,75),25,cv2.BORDER_DEFAULT)

            print(test_loader.fine_coarse_add)
            cv2.namedWindow('smap',cv2.WINDOW_NORMAL)
            cv2.imshow('smap',smap)

            target=target.squeeze().cpu().numpy()
            cv2.namedWindow('target',cv2.WINDOW_NORMAL)
            cv2.imshow('target',target)

            cv2.waitKey(0)

def main():
    parser=argparse.ArgumentParser()
    np.random.seed(12)

    # dataset type
    parser.add_argument('--test_dataset',type=str,default='osie')

    # gpu
    parser.add_argument('--gpu',default=True,action='store_true')

    # model dir
    parser.add_argument('--model_dir', type=str, default='src/salicon_model.pth')

    args=parser.parse_args()

    # Get dataloader (test)
    if args.test_dataset=='mit1003':
        args.test_dataset_dir='/mit1003_dataset'
        args.test_img_dir='ALLSTIMULI'
        args.test_label_dir='ALLFIXATIONMAPS'
    elif args.test_dataset=='osie':
        args.test_dataset_dir='/osie_dataset/data'
        args.test_img_dir='stimuli'
        args.test_label_dir='fixation_maps'
    else:
        raise NotImplemented

    test_dataset_dir=os.path.abspath('..')+args.test_dataset_dir
    dataloaders=getTest_loader(test_dataset_dir,args.test_img_dir,args.test_label_dir)

    # init the model
    model_weight=os.path.join(os.path.abspath('..'),args.model_dir)

    device=torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    model_trained=Salicon()
    model_trained.load_state_dict(torch.load(model_weight))
    model_trained.to(device)

    print("Begin test, Device: {}".format(device))

    # test the model
    test(model_trained,device,dataloaders)

if __name__ == '__main__':
    main()