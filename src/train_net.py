

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms

import os
import argparse
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"..")))

from src.data_utils import getTrainVal_loader
from src.salicon_model import Salicon


def train_model(model,device,dataloaders,criterion,optimizer,num_epochs,check_point=1):
    model.train()

    for epoch in range(num_epochs):

        print("Epoch {}/{}".format(epoch+1,num_epochs))
        print("-"*20)

        for phase in ['train','val']:
            running_loss=0
            if phase=='train':
                model.train()
            else:
                model.eval()

            cnt=0
            for inputs,labels in dataloaders[phase]:
                fine_img,coarse_img=inputs
                fine_img=fine_img.to(device)
                coarse_img=coarse_img.to(device)
                labels=labels.to(device)

                with torch.autograd.set_grad_enabled(phase=='train'):
                    outputs=model(fine_img,coarse_img)
                    loss=criterion(outputs,labels)
                if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                cnt=cnt+1
                if cnt==len(dataloaders[phase]):
                    print("\r {} Complete: {:.2f}".format(phase,cnt/len(dataloaders[phase])),end='\n')
                else:
                    print("\r {} Complete: {:.2f}".format(phase,cnt/len(dataloaders[phase])),end="")

                running_loss+=loss.item()*fine_img.size(0)

            epoch_loss=running_loss/len(dataloaders[phase])

            print("{} Loss: {}".format(phase,epoch_loss))

        if num_epochs%check_point==0:
            torch.save(model.state_dict(),'salicon_model.pth')


def main():
    parser=argparse.ArgumentParser()
    np.random.seed(12)

    # dataset dir
    parser.add_argument('--train_dataset_dir',type=str,default='/osie_dataset/data')
    parser.add_argument('--train_img_dir', type=str, default='stimuli')
    parser.add_argument('--train_label_dir', type=str, default='fixation_maps')

    # train parameters
    parser.add_argument('--batch_size',type=int,default=1,help='input batch size for training')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate for training')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for training')
    parser.add_argument('--decay', type=float, default=0.0005, help='weight decay for training')
    parser.add_argument('--epochs', type=int, default=500, help='epoch for training')
    parser.add_argument('--checkpoint', type=int, default=1, help='checkpoint for save model')

    # gpu
    parser.add_argument('--gpu',default=True,action='store_true')


    args=parser.parse_args()


    # Get dataloader (train and val)
    train_dataset_dir=os.path.abspath('..')+args.train_dataset_dir
    dataloaders=getTrainVal_loader(train_dataset_dir,args.train_img_dir,args.train_label_dir)

    # init the model
    device=torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    model_ft=Salicon()
    model_ft.to(device)
    optimizer_ft=optim.SGD(model_ft.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.decay,nesterov=True)
    criterion_ft=nn.BCELoss()
    print("Begin train, Device: {}".format(device))
    # train model
    train_model(model_ft, device, dataloaders, criterion_ft, optimizer_ft, num_epochs=args.epochs,check_point=args.checkpoint)


if __name__ == '__main__':
    main()