import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import os

from models import CNN
from dataset import ImgPatches, ConcatDatasets
from loss import ssim
from transforms import AddRandomScatter

parser = argparse.ArgumentParser()
parser.add_argument('model', type=str,
        help='path of the trained model')
parser.add_argument('test_dir', type=str,
        help='directory containing the testing dataset')
parser.add_argument('--patch-size', type=int, default=64,
        help='size of the sliding window for patch extraction (default 64)')
parser.add_argument('--patch-stride', type=int, default=64,
        help='stride of the sliding window for patch extraction (default 64)')
parser.add_argument('--out-dir', type=str, default='results',
        help='output directory where results are saved (default results)')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    args = parser.parse_args()

    test_data = ImgPatches(
            root_dir = args.test_dir,
            patch_size = args.patch_size,
            stride = args.patch_stride,
            transform = transforms.CenterCrop(900),
    )
    
    test_data_noisy = ImgPatches(
            root_dir = args.test_dir,
            patch_size = args.patch_size,
            stride = args.patch_stride,
            transform = transforms.CenterCrop(900),
            patch_transform = AddRandomScatter(51,(15,20),(0.5,0.7),'uniform')
    )

    test_loader = DataLoader(
            ConcatDatasets(test_data[:10], 
                test_data_noisy[:10]),
            batch_size=1,
            shuffle = False,
    )

    model = CNN()
    model.load_state_dict(torch.load(args.model))
    model.eval()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    with torch.no_grad():
        for batch, (true, noisy) in enumerate(test_loader):
            pred = model(noisy).detach()
            ssim1 = torch.mean(ssim(true,noisy,11,1.5))
            ssim2 = torch.mean(ssim(true,pred,11,1.5))
            print(f'Test {batch}; SSIM: {ssim1:>7f} {ssim2:>7f}')

            pred = pred.squeeze().numpy()*255
            noisy = noisy.squeeze().numpy()*255
            true = true.squeeze().numpy()*255
            output = np.concatenate((true, noisy, pred), axis=1)
            cv2.imwrite(f'results/pred{batch:>03d}.png', output)

if __name__ == '__main__':
    main()
