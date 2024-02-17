import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ssim_torch import ssim

def torch_psnr(img, ref):  # input [28,256,256]
    img = (img*65536).round()
    ref = (ref*65536).round()
    nC = img.shape[0]
    psnr = 0
    for i in range(nC):
        mse = torch.mean((img[i, :, :] - ref[i, :, :]) ** 2)
        psnr += 10 * torch.log10((65535*65535)/mse)
    return psnr / nC

def torch_ssim(img, ref):  # input [28,256,256]
    return ssim(torch.unsqueeze(img, 0), torch.unsqueeze(ref, 0))