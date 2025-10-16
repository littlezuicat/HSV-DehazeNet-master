import torch,os,sys,torchvision,argparse
import torchvision.transforms as tfs
import time,math
import numpy as np
from torch.backends import cudnn
from torch import optim
import torch,warnings
from torch import nn
import torchvision.utils as vutils
warnings.filterwarnings('ignore')


parser=argparse.ArgumentParser()
parser.add_argument('--steps',type=int,default=30000)
parser.add_argument('--device',type=str,default='Automatic detection')
parser.add_argument('--resume',type=bool,default=False)
parser.add_argument('--eval_step',type=int,default=100)
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--model_dir',type=str,default='./trained_models/')
parser.add_argument('--net',type=str,default='HSVDehazeNet')
parser.add_argument('--bs',type=int,default=2,help='batch size')
parser.add_argument('--workers',type=int,default=0,help='num of workers')
parser.add_argument('--crop',action='store_true')
parser.add_argument('--crop_size',type=int,default=256,help='Takes effect when using --crop ')
parser.add_argument('--path',type=str,default=r'C:\Users\33913\Desktop\dataset',help='Dataset location')


opt=parser.parse_args()

def get_dataset_name():
    datasets = os.listdir(opt.path)
    print(f"Currently, the following datasets are available in {opt.path}: Please enter the number and press Enter\n")
    for idx, dataset in enumerate(datasets):
        print(idx+1, dataset)
    print('-->', end='')
    dataset_name = datasets[int(input())-1]
    
    return dataset_name

opt.dataset_name = get_dataset_name()
opt.device='cuda' if torch.cuda.is_available() else 'cpu'
if opt.device=='cpu':
	for i in range(5):
		print("You are using CPU to train!!!!!!")
		time.sleep(0.5)
opt.model_name=opt.dataset_name+'_'+opt.net
opt.model_dir=opt.model_dir+opt.model_name+'.pk'
log_dir='logs/'+opt.model_name



if not os.path.exists('trained_models'):
	os.mkdir('trained_models')
if not os.path.exists('logs'):
	os.mkdir('logs')
if not os.path.exists('result'):
	os.mkdir('result')
if not os.path.exists(f"result/{opt.model_name}"):
	os.mkdir(f'result/{opt.model_name}')
if not os.path.exists(log_dir):
	os.mkdir(log_dir)

