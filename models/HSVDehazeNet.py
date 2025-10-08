import torch
import torch.nn as nn
from models.HSVNet import HSVNet
from models.RGBNet import RGBNet
from torch.nn import functional as FF



class AFF(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.conv_in = nn.ModuleList([nn.Conv2d(dim, dim, kernel_size=1) for i in range(2)])
        self.conv_out = nn.ModuleList([nn.Conv2d(dim, dim, kernel_size=1) for i in range(2)])
        self.batch_norm64 = nn.BatchNorm2d(dim)
        self.batch_norm8 = nn.BatchNorm2d(dim)
    def forward(self, x, y):
        f = x + y
        f1 = FF.gelu(self.batch_norm8(self.conv_in[0](self.GAP(f))))
        f1 = self.batch_norm64(self.conv_out[0](f1))
        
        f2 = FF.gelu(self.batch_norm8(self.conv_in[1](f)))
        f2 = self.batch_norm64(self.conv_out[1](f2))
        
        sum = FF.sigmoid(f1 + f2)
        
        out = x*sum + y * (1-sum)
        return out

class HSVDehazeNet(nn.Module):
    def __init__(self, dim = 32):
        """return out_4x, out_2x, out_1x, H, S, V, A, output"""
        super().__init__()
        self.dim = dim
        self.RGBPath = RGBNet(dim=32)
        self.HSVPath = HSVNet(dim=32)
        self.AFF = AFF(dim=3)
        self.conv1 = nn.Conv2d(3, self.dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(3, self.dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(self.dim, 3, kernel_size=3, padding=1)
        self.output = []
    def forward(self, x):
        # RGBPath
        [out_4x, out_2x, out_1x] = self.RGBPath(x)
        # HSVPath
        out_1x_HSV, H, S, V, A = self.HSVPath(x)
        # HueCalibrationModule
        
        # SKFF
        out = self.AFF(out_1x, out_1x_HSV)
        
        return out_4x, out_2x, out_1x, H, S, V, A, out
    