import numpy as np
import math
import os
import sys
from torchvision import datasets,transforms
from torch.utils.data import Dataset,DataLoader,SubsetRandomSampler
from PIL import Image

vgg16_mean=np.array([123.68,116.779,103.939])/255.
vgg16_std=np.array([0.229,0.224,0.225])

class SaliconDataset(Dataset):

    def __init__(self,dataset_root,stimuli_dir,fixation_maps_dir,transform=None,use_cache=False,type='train'):

        super(SaliconDataset,self).__init__()

        self.dataset_root=dataset_root

        self.stimuli_dir=stimuli_dir

        self.fixation_maps_dir=fixation_maps_dir

        self.image_names = [f for f in os.listdir(os.path.join(dataset_root, stimuli_dir)) if
                       os.path.isfile(os.path.join(dataset_root, stimuli_dir, f)) and not f.startswith('.')]

        self.transform=transform

        self.use_cache=use_cache

        self.lens=len(self.image_names)

    def __getitem__(self, index):
        fine_coarse_add=os.path.join(self.dataset_root,self.stimuli_dir,self.image_names[index])
        label_add=os.path.join(self.dataset_root,self.fixation_maps_dir,self.image_names[index].replace('.jpeg','.jpg'))
        self.fine_coarse_add=fine_coarse_add
        original_img=Image.open(fine_coarse_add)
        label=Image.open(label_add).convert('L')

        if self.transform is None:
            fine_img=coarse_img=original_img
        elif all([x in self.transform.keys() for x in ['fine','coarse','label']]):
            fine_img=self.transform['fine'](original_img)
            coarse_img=self.transform['coarse'](original_img)
            label=self.transform['label'](label)
        else:
            raise NotImplemented

        return [fine_img,coarse_img],label


    def __len__(self):
        return self.lens

def getTrainVal_loader(train_dataset_dir,img_dir,label_dir,shuffle=True,val_split=0.1):

    vgg16_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32) / 255.
    vgg16_std = np.array([0.229, 0.224, 0.225])

    data_transforms={
        'fine' : transforms.Compose([
            transforms.Resize((600,800),interpolation=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=vgg16_mean, std=vgg16_std),
        ]),
        'coarse':transforms.Compose([
            transforms.Resize((300,400),interpolation=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=vgg16_mean, std=vgg16_std),
        ]),
        'label': transforms.Compose([
            transforms.Resize((37, 50), interpolation=0),
            transforms.ToTensor(),
            # transforms.Normalize(mean=vgg16_mean, std=vgg16_std),
        ])
    }

    trainval_dataset = SaliconDataset(train_dataset_dir, img_dir, label_dir, transform=data_transforms)

    dataset_size=len(trainval_dataset)
    indices=list(range(dataset_size))
    split=int(np.floor(val_split*dataset_size))
    if shuffle:
        np.random.shuffle(indices)

    train_indices,val_indices=indices[split:],indices[:split]

    train_sampler=SubsetRandomSampler(train_indices)
    valid_sampler=SubsetRandomSampler(val_indices)

    train_loader=DataLoader(trainval_dataset,batch_size=1,sampler=train_sampler,num_workers=6)
    val_loader=DataLoader(trainval_dataset,batch_size=1,sampler=valid_sampler,num_workers=6)

    trainval_loaders={'train':train_loader,'val':val_loader}

    return trainval_loaders

def getTest_loader(test_dataset_dir,img_dir,label_dir,shuffle=True):

    vgg16_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32) / 255.
    vgg16_std = np.array([0.229, 0.224, 0.225])

    data_transforms={
        'fine' : transforms.Compose([
            transforms.Resize((600,800),interpolation=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=vgg16_mean, std=vgg16_std),
        ]),
        'coarse':transforms.Compose([
            transforms.Resize((300,400),interpolation=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=vgg16_mean, std=vgg16_std),
        ]),
        'label': transforms.Compose([
            # transforms.Resize((37, 50), interpolation=2),
            transforms.ToTensor(),
            # transforms.Normalize(mean=vgg16_mean, std=vgg16_std),
        ])
    }

    test_dataset = SaliconDataset(test_dataset_dir, img_dir, label_dir, transform=data_transforms)

    #test_loader=DataLoader(test_dataset,batch_size=1,num_workers=6)

    return test_dataset

if __name__ == '__main__':
    project_dir=os.path.abspath('..')
    # print(project_dir)
    mit1003_dir = project_dir+'/mit1003_dataset'

    osie_dir=project_dir+'/osie_dataset/data'

    data_transforms={
        'fine' : transforms.Compose([
            transforms.Resize((600,800),interpolation=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=vgg16_mean, std=vgg16_std),
        ]),
        'coarse':transforms.Compose([
            transforms.Resize((300,400),interpolation=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=vgg16_mean, std=vgg16_std),
        ]),
        'label': transforms.Compose([
            transforms.Resize((37, 50), interpolation=0),
            transforms.ToTensor(),
            # transforms.Normalize(mean=vgg16_mean, std=vgg16_std),
        ])
    }

    test_dataset=SaliconDataset(mit1003_dir,'ALLSTIMULI','ALLFIXATIONMAPS',transform=data_transforms)

    trainval_dataset=SaliconDataset(osie_dir,'stimuli','fixation_maps',transform=data_transforms)


    # divide trainval dataset to train dataset and val dataset
    shuffle_dataset=True
    val_split=0.1
    random_seed=42

    dataset_size=len(trainval_dataset)
    indices=list(range(dataset_size))
    split=int(np.floor(val_split*dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices,val_indices=indices[split:],indices[:split]

    train_sampler=SubsetRandomSampler(train_indices)
    valid_sampler=SubsetRandomSampler(val_indices)

    loader_train=DataLoader(trainval_dataset,batch_size=1,sampler=train_sampler,num_workers=6)
    loader_val=DataLoader(trainval_dataset,batch_size=1,sampler=valid_sampler,num_workers=6)

    input,output=(next(iter(loader_train)))
    fine_img,coarse_img=input
    print(fine_img.shape,coarse_img.shape)
    # import_train_data(mit1003_dir,'ALLSTIMULI','ALLFIXATIONMAPS',use_cache=False)
    # osie_dataset=project_dir+'/osie_dataset'
    # get_mit1003_data(mit1003_dir)



# def get_mit1003_data(path):
#     mit1003_stimuli=path+'/ALLSTIMULI'
#     mit1003_labels=path+'/ALLFIXATIONMAPS'
#     #mit1003_dir='./mit1003_dataset'
#     #test=os.path.exists(mit1003_dir)
#     #print(test)
#
# def import_train_data(dataset_root,stimuli_dir,fixation_maps_dir,use_cache=True):
#
#     if use_cache:
#         cache_dir=os.path.join(dataset_root,'__cache')
#         if not os.path.isdir(cache_dir):
#             os.mkdir(cache_dir)
#
#         if os.path.isfile(os.path.join(cache_dir,'data.npy')):
#             fine_image_data,coarse_image_data,label_data=np.load(os.path.join(cache_dir,'data.npy'))#np.load()
#             return fine_image_data,coarse_image_data,label_data
#
#
#     image_names=[f for f in os.listdir(os.path.join(dataset_root,stimuli_dir)) if os.path.isfile(os.path.join(dataset_root,stimuli_dir,f))]
#
#     fine_image_data=np.zeros((len(image_names),600,800,3),dtype=np.float32)
#     coarse_image_data = np.zeros((len(image_names), 300, 400, 3), dtype=np.float32)
#     label_data = np.zeros((len(image_names), 37, 50, 1), dtype=np.float32)
#
#     data_transforms={
#         'fine' : transforms.Compose([
#             transforms.Resize((600,800),interpolation=0),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=vgg16_mean, std=vgg16_std),
#         ]),
#         'coarse':transforms.Compose([
#             transforms.Resize((300,400),interpolation=0),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=vgg16_mean, std=vgg16_std),
#         ]),
#         'label': transforms.Compose([
#             transforms.Resize((37, 50), interpolation=0),
#             transforms.ToTensor(),
#             # transforms.Normalize(mean=vgg16_mean, std=vgg16_std),
#         ])
#     }
#
#
#     for i in range(len(image_names)):
#         fine_coarse_add=os.path.join(dataset_root,stimuli_dir,image_names[i])
#         label_add=os.path.join(dataset_root,fixation_maps_dir,image_names[i].replace('.jpeg','.jpg'))
#         # print(label_add)
#     #    img_fine=