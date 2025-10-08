"""
Author: Yi Ren
Email: renyi@buu.edu.cn
Affiliation: Beijing Union University, Beijing, China
"""
import torch, os, sys, torchvision, argparse
import torchvision.transforms as tfs
from metrics import psnr, ssim
from models.HSVDehazeNet import HSVDehazeNet
import time, math
import numpy as np
from torch.backends import cudnn
from torch import optim
import warnings
from torch import nn
import torchvision.utils as vutils
from option import opt, log_dir
from data_utils import *
from torchvision.models import vgg16
from tqdm import tqdm
from datetime import datetime
from loss import loss_fn
from torch.nn import functional as FF
# from torch.utils.tensorboard import SummaryWriter
# Import tensorboard

# writer = SummaryWriter(log_dir=log_dir)
warnings.filterwarnings('ignore')

print('log_dir :', log_dir)
print('model_name:', opt.model_name)

models_ = {
    'HSVDehazeNet': HSVDehazeNet()
}


start_time = time.time()
T = opt.steps

def lr_schedule_cosdecay(t, T, init_lr=opt.lr):
    lr = 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
    return lr

def train(net, loader_train, loader_test, optim):
    losses = []
    start_step = 0
    max_ssim = 0
    max_psnr = 0
    ssims = []
    psnrs = []

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'log_.txt')

    # Check if training continues from an existing model
    if opt.resume and os.path.exists(opt.model_dir):
        print(f'resume from {opt.model_dir}')
        ckp = torch.load(opt.model_dir)
        losses = ckp['losses']
        net.load_state_dict(ckp['model'])
        start_step = ckp['step']
        max_ssim = ckp['max_ssim']
        max_psnr = ckp['max_psnr']
        psnrs = ckp['psnrs']
        ssims = ckp['ssims']
        print(f'start_step: {start_step} start training ---')
    else:
        print('train from scratch *** ')
    with tqdm(range(start_step + 1, opt.steps + 1), unit='step', ncols=130) as steps:
        with open(log_file, 'a') as f:
            for step in steps:  
                optim.zero_grad()              
                lr = opt.lr
                steps.set_description(f"model:{opt.model_name} step:[{step}]/{opt.steps}")
            
                lr = lr_schedule_cosdecay(step, T)
                for param_group in optim.param_groups:
                    param_group["lr"] = lr  

                x, y= next(iter(loader_train))
                x = x.to(opt.device)
                y = y.to(opt.device)
                
                out_4x, out_2x, out_1x, H, S, V, A, output = net(x)
                
                loss = loss_fn(out_4x, out_2x, out_1x, H, S, V, A, output, y)
                loss.backward()
               
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.001)
                optim.step()
                
                losses.append(loss.item())
                if step % 50 == 0:
                    log_message = f'model:{opt.net} step: {step}, loss: {loss.item():.6f}, psnr: {max_psnr}, ssim: {max_ssim}, minute:{(time.time()-start_time)/60:.2f}'
                    f.write(log_message + '\n')

                steps.set_postfix(loss=np.mean(losses), lr=lr, bs=opt.bs)

                # Test the model and save the best results
                if step % opt.eval_step == 0:
                    with torch.no_grad():
                        ssim_eval, psnr_eval = test(net, loader_test, max_psnr, max_ssim, step)
                    print(f'\nstep :{step} | ssim:{ssim_eval:.4f} | psnr:{psnr_eval:.4f}')
                    ssims.append(ssim_eval)
                    psnrs.append(psnr_eval)

                    if ssim_eval > max_ssim and psnr_eval > max_psnr:
                        max_ssim = max(max_ssim, ssim_eval)
                        max_psnr = max(max_psnr, psnr_eval)
                        torch.save({
                            'step': step,
                            'max_psnr': max_psnr,
                            'max_ssim': max_ssim,
                            'ssims': ssims,
                            'psnrs': psnrs,
                            'losses': losses,
                            'model': net.state_dict()
                        }, opt.model_dir)
                    print(f'\n model saved at {opt.model_dir}, step: {step} | max_psnr: {max_psnr:.4f} | max_ssim: {max_ssim:.4f}')
                    

def test(net, loader_test, max_psnr, max_ssim, step):
    net.eval()
    torch.cuda.empty_cache()
    ssims = []
    psnrs = []
    outputs = []
    output_dir = os.path.join('result', opt.model_name)
    for i, (inputs, targets) in enumerate(loader_test):
            inputs = inputs.to(opt.device)
            targets = targets.to(opt.device)
            _, _, _, _, _, _, _, output = net(inputs)
            ssim1 = ssim(output.clamp(0,1), targets).item()
            psnr1 = psnr(output.clamp(0,1), targets)
            
            ssims.append(ssim1)
            psnrs.append(psnr1)
            outputs.append(output)
            # Save images
            ts = torch.squeeze(output.clamp(0, 1).cpu())
    if(np.mean(ssims) > max_ssim and np.mean(psnrs) > max_psnr):
        for i, output in enumerate(outputs):
            ts = torch.squeeze(output.clamp(0, 1).cpu())
            vutils.save_image(ts, output_dir + f'/{str(i)}_{opt.model_name}.png')
        print('new pic saved')
    return np.mean(ssims), np.mean(psnrs)

if __name__ == "__main__":
    net = models_[opt.net]
    net = net.to(opt.device)

    net.train()
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    

    optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08)
    optimizer.zero_grad()
    
    train(net, loader_train, loader_test, optimizer)
