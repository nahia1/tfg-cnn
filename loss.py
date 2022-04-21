import torch
import torch.nn as nn
from torchvision.transforms.functional import gaussian_blur
import numpy as np

def ssim(img1, img2, kernel_size, kernel_sigma, k1=0.01, k2=0.03):
    c1 = k1**2
    c2 = k2**2
    mu1 = gaussian_blur(img1, kernel_size, kernel_sigma)
    mu2 = gaussian_blur(img2, kernel_size, kernel_sigma)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu12 = mu1*mu2

    sigma1_sq = gaussian_blur(img1 * img1, kernel_size, kernel_sigma) - mu1_sq
    sigma2_sq = gaussian_blur(img2 * img2, kernel_size, kernel_sigma) - mu2_sq
    sigma12 = gaussian_blur(img1 * img2, kernel_size, kernel_sigma) - mu12

    ssim_map = ((2*mu12 + c1)*(2*sigma12 + c2)) / ((mu1_sq + mu2_sq + c1)*
                    (sigma1_sq + sigma2_sq + c2))

    return torch.clamp(ssim_map, min=0, max=1)

class SSIMLoss(nn.Module):
    def __init__(self, kernel_size=11, kernel_sigma=1.5, reduction = 'mean'):
        super(SSIMLoss, self).__init__()
        self.kernel_size = kernel_size
        self.kernel_sigma = kernel_sigma
        self.reduction = reduction
    def forward(self, input, target):
        loss = (1 - ssim(input, target, self.kernel_size, self.kernel_sigma))/2

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        else:
            pass

        return loss

class L2_SSIMLoss(nn.Module):
    def __init__(self, kernel_size=11, kernel_sigma=1.5, a=0.5, reduction = 'mean'):
        super(L2_SSIMLoss, self).__init__()
        self.kernel_size = kernel_size
        self.kernel_sigma = kernel_sigma
        self.a = a
        self.reduction = reduction

    def forward(self, input, target):
        l2_loss = (input - target).pow(2)
        ssim_loss = (1 - ssim(input, target, self.kernel_size, self.kernel_sigma))/2
        loss = self.a*l2_loss + (1-self.a)*ssim_loss

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        else:
            pass

        return loss
