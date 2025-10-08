from math import exp
import math
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from math import exp
import math
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from  torchvision.transforms import ToPILImage
from option import opt



def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
    img1=torch.clamp(img1,min=0,max=1)
    img2=torch.clamp(img2,min=0,max=1)
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)
def psnr(pred, gt):
    pred=pred.clamp(0,1).cpu().numpy()
    gt=gt.clamp(0,1).cpu().numpy()
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10( 1.0 / rmse)



def denormalize_tensor(tensor):
    if opt.trainset == 'haze4k_train':
        mean=[0.5755, 0.5443, 0.5320]
        std=[0.2299, 0.2393, 0.2500]
    elif opt.trainset == "Ohaze_train":
        mean = [0.4757, 0.5087, 0.5682]
        std = [0.1034, 0.1062, 0.1155]
    else:
        raise "The mean and standard deviation of the dataset do not exist. Please manually add them in data_utils.py"
    """
    Denormalize tensor data
    :param tensor: Input image tensor, shape (B, C, H, W)
    :param mean: RGB mean (3,)
    :param std: RGB standard deviation (3,)
    :return: Denormalized tensor
    """
    new_tensor = tensor.clone()
    for i in range(3):  # Denormalize each channel
        new_tensor[:, i, :, :] = new_tensor[:, i, :, :] * std[i] + mean[i]
    return new_tensor

def rgb_to_hsv_fn(rgb_tensor):
    """
    Convert RGB Tensor of shape (B, C, H, W) to HSV Tensor
    :param rgb_tensor: Input RGB image tensor, shape (B, C, H, W)
    :return: Converted HSV image tensor, shape (B, C, H, W), range is 0-1 (not normalized anymore)
    """

    "Note: This function will revert normalized data back to original values!"
    # Convert PyTorch Tensor to Numpy array, note that it needs to be reshaped to (B * H * W, 3)
    denormalize_tensor(rgb_tensor)  # At this point, the range is 0-1
    rgb_tensor = rgb_tensor.permute(0, 2, 3, 1)  # Convert to (B, H, W, C)

    rgb_np = rgb_tensor.cpu().detach().numpy().astype(np.uint8)  # At this point, it has been converted to the range of 0-255

    # Convert each image using OpenCV from RGB to HSV
    hsv_np = np.zeros_like(rgb_np, dtype=np.float32)
    for i in range(b):
        hsv_np[i] = cv2.cvtColor(rgb_np[i], cv2.COLOR_RGB2HSV)  # Convert each image

    # Normalize each channel of HSV image
    # Hue H normalization: 0 - 179 -> 0 - 1
    hsv_np[:, :, :, 0] = hsv_np[:, :, :, 0] / 179.0  # Normalize H
    # Saturation S and Brightness V normalization: 0 - 255 -> 0 - 1
    hsv_np[:, :, :, 1] = hsv_np[:, :, :, 1] / 255.0  # Normalize S
    hsv_np[:, :, :, 2] = hsv_np[:, :, :, 2] / 255.0  # Normalize V

    # Convert Numpy array back to PyTorch Tensor and return
    hsv_tensor = torch.from_numpy(hsv_np).permute(0, 3, 1, 2)  # Convert back to (B, C, H, W)
    if torch.cuda.is_available():
       hsv_tensor = hsv_tensor.cuda()
    return hsv_tensor

def hsv_to_rgb_fn(hsv_tensor):
    """
    Convert HSV Tensor of shape (B, C, H, W) to RGB Tensor
    :param hsv_tensor: Input HSV image tensor, shape (B, C, H, W), H, S, V range [0, 1]
    :return: Converted RGB image tensor, shape (B, C, H, W), RGB range [0, 1]
    """

    # Extract HSV channels

    b, c, h, w = hsv_tensor.shape
    hsv_tensor = hsv_tensor.permute(0, 2, 3, 1)  # Convert to (B, H, W, C)
    hsv_tensor = hsv_tensor.clone()
    hsv_np = hsv_tensor.cpu().detach().numpy()

    # Reverse normalization for each channel of HSV image
    # H reverse normalization: [0, 1] -> [0, 179]
    hsv_np[:, :, :, 0] = hsv_np[:, :, :, 0] * 179.0
    # S and V reverse normalization: [0, 1] -> [0, 255]
    hsv_np[:, :, :, 1] = hsv_np[:, :, :, 1] * 255.0
    hsv_np[:, :, :, 2] = hsv_np[:, :, :, 2] * 255.0

    # Clip the values to the valid range
    hsv_np[:, :, :, 0] = np.clip(hsv_np[:, :, :, 0], 0, 179)  # H only clip to [0, 179]
    hsv_np = np.clip(hsv_np, 0, 255).astype(np.uint8)

    # Convert each image using OpenCV from HSV to RGB
    rgb_np = np.zeros_like(hsv_np, dtype=np.uint8)
    for i in range(b):
        rgb_np[i] = cv2.cvtColor(hsv_np[i], cv2.COLOR_HSV2RGB)  # Convert each image

    # Convert Numpy array back to PyTorch Tensor, and return
    rgb_tensor = torch.from_numpy(rgb_np).permute(0, 3, 1, 2) / 255  # Convert back to (B, C, H, W)
    if torch.cuda.is_available():
        rgb_tensor = rgb_tensor.cuda()
    return rgb_tensor



def rgb_to_hsv(tensor):
    """
    Convert RGB tensor in BCHW format to HSV format
    
    Arguments:
        tensor (torch.Tensor): Input RGB tensor, format BCHW (B: batch size, C: 3, H: height, W: width)
    
    Returns:
        torch.Tensor: Converted HSV tensor, format still BCHW
    """
    tensor = denormalize_tensor(tensor)
    
    # Extract RGB channels
    r, g, b = tensor[:, 0:1, :, :], tensor[:, 1:2, :, :], tensor[:, 2:3, :, :]
    
    # Calculate the maximum, minimum, and their difference
    max_val, _ = torch.max(tensor, dim=1, keepdim=True)  # Maximum value
    min_val, _ = torch.min(tensor, dim=1, keepdim=True)  # Minimum value
    delta = max_val - min_val  # Difference between maximum and minimum
    
    # Initialize hue, saturation, and value
    h = torch.zeros_like(max_val)  # Initialize hue channel
    s = torch.zeros_like(max_val)  # Initialize saturation channel
    v = max_val  # Value is the maximum value
    
    # Avoid division by zero, only calculate hue when max and min values are different (i.e., there is color)
    mask = delta != 0
    # print(v == r, v == g, v== b)
    # Calculate hue (H)
    # If R is the maximum value
    if(r == max_val).any():
        # print('r is the best')
        h[mask] = 60 * ((g[mask] - b[mask]) / delta[mask])
    # If G is the maximum value
    elif(g == max_val).any():
        # print('g is the best')
        h[mask] = 60 * ((b[mask] - r[mask]) / delta[mask] + 2) 
    elif(b == max_val).any():
        # print('b is the best')
    # If B is the maximum value
        h[mask] = 60 * ((r[mask] - g[mask]) / delta[mask] + 4)
    # Calculate saturation (S)
    s[mask] = delta[mask] / max_val[mask]  # Saturation is the ratio of the difference to the maximum value
    # print(22222222222222222222, torch.max(h), torch.min(h))
    # Combine H, S, V channels into one HSV tensor
    hsv_tensor = torch.cat([h, s, v], dim=1)  # Concatenate to form BCHW formatted HSV tensor

    return hsv_tensor


def hsv_to_rgb1(hsv_tensor):
    """
    Convert HSV tensor in BCHW format to RGB format
    
    Arguments:
        hsv_tensor (torch.Tensor): Input HSV tensor, format BCHW (B: batch size, C: 3, H: height, W: width)
            - h (hue): range [0, 360)
            - s (saturation): range [0, 1]
            - v (value): range [0, 1]
    
    Returns:
        torch.Tensor: Converted RGB tensor, format still BCHW
            - r, g, b (RGB channels): range [0, 1]
    """
    
    # Extract HSV channels
    h, s, v = hsv_tensor[:, 0:1, :, :], hsv_tensor[:, 1:2, :, :], hsv_tensor[:, 2:3, :, :]

    # Calculate the range of hue (0 <= H < 360), and convert to ratio in [0, 1]
    h = h / 60.0  # Convert hue degrees to [0, 6) ratio
    i = torch.floor(h)  # Hue range [0, 6), indicates which part of the color wheel
    f = h - i  # Remainder part of hue [0, 1) fraction

    p = v * (1 - s)  # Calculate the product of value and saturation, yields the low-saturation color component
    q = v * (1 - f * s)  # Calculate transition color value between red and green
    t = v * (1 - (1 - f) * s)  # Calculate transition color value between blue and green
    
    # Initialize RGB channels
    r, g, b = torch.zeros_like(v), torch.zeros_like(v), torch.zeros_like(v)

    # Select different RGB values based on the hue range i
    # Assign color components based on different values of i
    r[i == 0] = v[i == 0]
    g[i == 0] = t[i == 0]
    b[i == 0] = p[i == 0]

    r[i == 1] = q[i == 1]
    g[i == 1] = v[i == 1]
    b[i == 1] = p[i == 1]

    r[i == 2] = p[i == 2]
    g[i == 2] = v[i == 2]
    b[i == 2] = t[i == 2]

    r[i == 3] = p[i == 3]
    g[i == 3] = q[i == 3]
    b[i == 3] = v[i == 3]

    r[i == 4] = t[i == 4]
    g[i == 4] = p[i == 4]
    b[i == 4] = v[i == 4]

    r[i == 5] = v[i == 5]
    g[i == 5] = p[i == 5]
    b[i == 5] = q[i == 5]

    # Combine RGB channels
    rgb_tensor = torch.cat([r, g, b], dim=1)
    
    return rgb_tensor

if __name__ == "__main__":
    pass
