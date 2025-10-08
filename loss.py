import torch
import numpy
from metrics import *
import torch.nn.functional as FF
import kornia

def loss_fn(out_4x, out_2x, out_1x, H, S, V, A, output, y):
    """input: out_4x, out_2x, out_1x, H, S, V, A, output, y
    return: loss"""
    H_GT = kornia.color.rgb_to_hsv(y)[:, 0:1, :, :]
    S_GT = kornia.color.rgb_to_hsv(y)[:, 1:2, :, :]
    V_GT = kornia.color.rgb_to_hsv(y)[:, 2:3, :, :]
    y2 = F.interpolate(y, scale_factor=0.5, mode='bilinear')
    y4 = F.interpolate(y, scale_factor=0.25, mode='bilinear')
    l1 = FF.l1_loss(out_4x, y4)
    l2 = FF.l1_loss(out_2x, y2)
    l3 = FF.l1_loss(out_1x, y)
    loss_content = l1+l2+l3

    label_fft1 = torch.fft.fft2(y4, dim=(-2,-1))
    label_fft1 = torch.stack((label_fft1.real, label_fft1.imag), -1)

    pred_fft1 = torch.fft.fft2(out_4x, dim=(-2,-1))
    pred_fft1 = torch.stack((pred_fft1.real, pred_fft1.imag), -1)

    label_fft2 = torch.fft.fft2(y2, dim=(-2,-1))
    label_fft2 = torch.stack((label_fft2.real, label_fft2.imag), -1)

    pred_fft2 = torch.fft.fft2(out_2x, dim=(-2,-1))
    pred_fft2 = torch.stack((pred_fft2.real, pred_fft2.imag), -1)

    label_fft3 = torch.fft.fft2(y, dim=(-2,-1))
    label_fft3 = torch.stack((label_fft3.real, label_fft3.imag), -1)

    pred_fft3 = torch.fft.fft2(out_1x, dim=(-2,-1))
    pred_fft3 = torch.stack((pred_fft3.real, pred_fft3.imag), -1)

    f1 = FF.l1_loss(pred_fft1, label_fft1)
    f2 = FF.l1_loss(pred_fft2, label_fft2)
    f3 = FF.l1_loss(pred_fft3, label_fft3)
    loss_fft = f1+f2+f3

    loss_hsv = FF.l1_loss(H, H_GT)+FF.l1_loss(S, S_GT)+FF.l1_loss(V, V_GT)+FF.l1_loss(A, S-V)
    loss = loss_content + 0.1 * loss_fft + 0.5 * loss_hsv
    return loss