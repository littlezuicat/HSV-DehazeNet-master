import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
from torch.nn import functional as FF_nn
import os,sys, re
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import random
from PIL import Image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from metrics import *
from option import opt
from tqdm import tqdm

BS=opt.bs
crop_size='whole_img'
if opt.crop:
    crop_size=opt.crop_size

class HSVDataset(data.Dataset):
    def __init__(self, path, train, size=crop_size):
        super().__init__()
        self.size = size
        print('crop size', size)
        self.train = train
        self.format = format
        print('Loading dataset into memory, please wait — larger datasets may take longer')
        # Preload dataset into memory
        haze_imgs_dir = sorted(os.listdir(os.path.join(path, 'hazy')))
        name = opt.dataset_name.lower().replace('-', '').replace('_', '')
        if any(haze in name for haze in ['ohaz', 'nhhaz', 'ihaz', 'densehaz']):
            
            clear_imgs_dir = sorted(os.listdir(os.path.join(path, 'GT')))
            self.haze_imgs = [
                Image.open(os.path.join(path, 'hazy', img)).convert("RGB")
                for img in haze_imgs_dir if img.endswith(('.jpg', '.png', 'JPG'))
            ]
            
            self.clear_imgs = [
                Image.open(os.path.join(path, 'GT', img)).convert("RGB")
                for img in clear_imgs_dir if img.endswith(('.jpg', '.png', 'JPG'))
            ]

        else:
            self.haze_imgs = []
            self.clear_imgs = []
            for hazy_img in tqdm(haze_imgs_dir):
                name, extension = os.path.splitext(hazy_img)
                clear_img =  re.match(r'\d+', name)[0]+extension
                self.clear_imgs.append(Image.open(os.path.join(path, 'GT', clear_img)).convert("RGB"))
                self.haze_imgs.append(Image.open(os.path.join(path, 'hazy', hazy_img)).convert("RGB"))
        print(f"[Dataset loaded successfully]\nLoaded {len(self.haze_imgs)} hazy images and {len(self.clear_imgs)} GT images into memory.\nNote: Loading the entire dataset into memory can significantly speed up training; high memory usage is expected.")

    def __getitem__(self, index):
        haze = self.haze_imgs[index]  # Directly retrieved from memory
        clear = self.clear_imgs[index]  # Directly retrieved from memory
        # print(haze)
        if not isinstance(self.size, str):
            i, j, h, w = tfs.RandomCrop.get_params(haze, output_size=(self.size, self.size))
            haze = FF.crop(haze, i, j, h, w)
            clear = FF.crop(clear, i, j, h, w)
        
        haze, clear= self.augData(haze, clear)
        return haze, clear
    
    def augData(self, data, target):
        
        if self.train:
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            
            data = tfs.RandomHorizontalFlip(rand_hor)(data)
            target = tfs.RandomHorizontalFlip(rand_hor)(target)
            
            if rand_rot:
                data = FF.rotate(data, 90 * rand_rot)
                target = FF.rotate(target, 90 * rand_rot)
        
            
        if not self.train:
            shaper = tfs.Resize((256, 256))
            data = shaper(data)
            target = shaper(target)
        data = tfs.ToTensor()(data)
        # data = tfs.Normalize(mean=mean, std=std)(data)
        target = tfs.ToTensor()(target)
        
            
        return data, target

    def __len__(self):
        return len(self.haze_imgs)



path=opt.path

loader_train=DataLoader(dataset=HSVDataset(path+f'/{opt.dataset_name}/train',train=True,size=crop_size),batch_size=opt.bs,shuffle=True, num_workers=opt.workers)
loader_test=DataLoader(dataset=HSVDataset(path+f'/{opt.dataset_name}/test',train=False,size='whole image'),batch_size=1,shuffle=False, num_workers=opt.workers)



if __name__ == "__main__":
    pass
