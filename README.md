# SALICON pytorch

This repository contains the code to train and run SALICON with pytorch. In our implementation we follow the original [CVPR'15 paper](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Huang_SALICON_Reducing_the_ICCV_2015_paper.pdf)

- [Introduction](##Introduction)
- [Prerequisites](##Prerequisites)
- [Instruciton](##Instruciton)

## Introduction


The implementation details of SALICON is based on [Kotseruba's work](https://github.com/ykotseruba/SALICONtf), whichs is implemented with tensorflow

More detalis please see the [original paper](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Huang_SALICON_Reducing_the_ICCV_2015_paper.pdf)


## Prerequisites

- Python>=3.5.0
- Pytorch>=1.5.0
- torchvision>=0.6.0
- opencv3>=3.1.0
- numpy>=1.14.2

## Instruciton



### Download Dataset



Download OSIE dataset if you want to train SALICON.

```
cd osie_dataset
sh download_osie_dataset.sh
```

Download MIT1003 dataset used for evaluation (optional).

```
cd mit1003_dataset
sh download_mit1003.sh
```


### Train



1 . Train a model with default parameters.

```
python3 train_net.py
```
2 . You can change the tranning parameters according to your needs, such as whether to use GPU, etc.


```
# use gpu
python3 train_net.py --gpu --batch_size 1 --lr 0.1 --momentum 0.9 --decay 0.0005 --epochs 500 --checkpoint 1
# use cpu only
python3 train_net.py --batch_size 1 --lr 0.1 --momentum 0.9 --decay 0.0005 --epochs 500 --checkpoint 1
```



### Test



1 . Test the model with default dataset (osie) .

```
python3 test.py
```

2 . Test the model with optional parameters, such as selecting the mit1003 dataset, whether to use gpu, etc.

```
python3 test.py --test_dataset mit1003 --gpu
```

### Download the Model weights
The model weights must be at the same level as test.py in the src directory

1. Baidu NetDisk
```
https://pan.baidu.com/s/1IdWwChDfLmOpRZfyABbdIQ 
password: 6wpu
```

2. ecloud NetDisk
```
https://cloud.189.cn/t/ZZrumuBFnyia
password: 5jzf
```

3. Google Drive
```
https://drive.google.com/file/d/1hxa1rlm94cV_dABu-1B3Jg3cQ9j_f3-P/view?usp=sharing
```



## Author



* **Zhengxi Hu**

Please raise an issue or send email to hzx@mail.nankai.edu.cn if there are any issues running the code.
