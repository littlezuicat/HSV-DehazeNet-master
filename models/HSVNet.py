import torch
import torch.nn as nn
import torch.nn.functional as FF
import kornia
from models.RGBNet import RGBNet as RGBNet

class HCM(nn.Module): # Hue Correction Module
    def __init__(self, dim):
        super().__init__()

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.gate_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.act = nn.GELU()
        self.group_norm = nn.GroupNorm(dim//8, dim)
    def forward(self, x):
        res1 = self.conv1(x)
        res2 = self.conv2(x)
        res = res1 + res2
        
        gate = self.gate_conv(x)
        gate = self.sigmoid(gate)
        
        res = res * gate
        
        output = self.act(self.group_norm(res))
        
        return output
    
class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv3_19 = nn.Conv2d(dim, dim, kernel_size=7, padding=9, groups=dim, dilation=3, padding_mode='reflect')
        self.conv3_13 = nn.Conv2d(dim, dim, kernel_size=5, padding=6, groups=dim, dilation=3, padding_mode='reflect')
        self.conv3_7 = nn.Conv2d(dim, dim, kernel_size=3, padding=3, groups=dim, dilation=3, padding_mode='reflect')
        self.group_norm_19 = nn.GroupNorm(dim//8, dim)
        self.group_norm_13 = nn.GroupNorm(dim//8, dim)
        self.group_norm_7 = nn.GroupNorm(dim//8, dim)
        self.CALayer = CA(dim)
        self.PALayer = PA(dim)
        self.SPALayer = SPA(dim)
        self.Out = nn.Conv2d(dim*3, dim, kernel_size=1)
    def forward(self, x):
        res1 = FF.gelu(self.group_norm_19(self.conv3_19(x)))
        res2 = FF.gelu(self.group_norm_13(self.conv3_13(x)))
        res3 = FF.gelu(self.group_norm_7(self.conv3_7(x)))
        res = res1 + res2 + res3
        res1 = self.CALayer(res)
        res2 = self.PALayer(res)
        res3 = self.SPALayer(res)
        output = FF.tanh(self.Out(torch.cat([res1, res2, res3], dim=1)))
        return output
    
class CA(nn.Module):
    def __init__(self, dim):
        super(CA, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(dim, dim//8, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(dim//8, dim, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        res = self.conv1(self.gap(x))
        res = self.conv2(self.act(res))
        weight = self.sigmoid(res)
        return x * weight
    

class SPA(nn.Module):
    def __init__(self, dim):
        super(SPA, self).__init__()
        self.Wv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=3 // 2, groups=dim, padding_mode='reflect')
        )
        self.Wg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        value = self.Wv(x)
        weight = self.Wg(x)
        return value * weight

class PA(nn.Module):
    def __init__(self, dim):
        super(PA, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim//8, kernel_size=1)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(dim//8, dim, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        res = self.conv1(x)
        res = self.conv2(self.act(res))
        weight = self.sigmoid(res)
        return x * weight
    


class HSVNet(nn.Module):
    def __init__(self, dim):
        """return out1x_RGB, out1x_HSV, H, S, V"""
        super().__init__()
        print('dim:', dim)
        self.ConvInHue = nn.Conv2d(1, dim, kernel_size=3, padding=1)
        self.ConOutHue = nn.Conv2d(dim, 1, kernel_size=3, padding=1)
        self.ConvIn = nn.Conv2d(3, dim, kernel_size=3, padding=1)
        self.HCMLayer = nn.Sequential(*[HCM(dim) for _ in range(1)])  # 5 color recovery modules
        self.blocksHSV = nn.Sequential(*[Block(dim) for _ in range(1)])  # 5 Blocks
        self.ConvOut = nn.Conv2d(dim, 3, kernel_size=3, padding=1)
        self.block_to_RGB = Block(dim)
        self.out = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )
    def forward(self, x):
        
        hsv = kornia.color.rgb_to_hsv(x)
        H = hsv[:, 0:1, :, :] # Hue channel
        S = hsv[:, 1:2, :, :] # Saturation channel
        V = hsv[:, 2:3, :, :] # Brightness channel
        H = self.HCMLayer(self.ConvInHue(H))
        H = self.ConOutHue(H)
        SVA = torch.cat([S, V, S-V], dim=1)
        SVA = self.ConvIn(SVA)
        SVA = self.blocksHSV(SVA)
        S = SVA[:, 0:1, :, :] # Saturation channel
        V = SVA[:, 1:2, :, :] # Brightness channel
        A = SVA[:, 2:3, :, :] # Fog density channel
        output_hsv = torch.cat([H, S, V], dim=1)
        output_rgb = kornia.color.hsv_to_rgb(output_hsv)
        
        return  output_rgb, H, S, V, A

