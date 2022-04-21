import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms

from dataset import ImgPatches, ConcatDatasets
from loss import L2_SSIMLoss, SSIMLoss
from transforms import AddRandomScatter
from models import CNN

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('train_dir', type=str,
        help='directory containing the training dataset')
parser.add_argument('--epochs', type=int, default=10, 
        help='number of training epochs (default 10)')
parser.add_argument('--batch-size', type=int, default=64,
        help='size of the mini-batches (default 64)')
parser.add_argument('--patch-size', type=int, default=64,
        help='size of the sliding window for patch extraction (default 64)')
parser.add_argument('--patch-stride', type=int, default=64,
        help='stride of the sliding window for patch extraction (default 64)')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(dataloader, model, loss_fn, optimiser):
    size = len(dataloader.dataset)
    model.train()

    for batch, (data_true, data_noisy) in enumerate(dataloader):
        data_true = data_true.to(device)
        data_noisy = data_noisy.to(device)
        pred = model(data_noisy)
        loss = loss_fn(pred, data_true)
        
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        loss, current = loss.item(), (batch + 1) * len(data_true)
        print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')

def validate(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for true, noisy in dataloader:
            true = true.to(device)
            noisy = noisy.to(device)
            pred = model(noisy)
            val_loss += loss_fn(pred, true).item()
    val_loss /= len(dataloader)
    return val_loss

def main():
    print(f'Using {device} device') 
    args = parser.parse_args()
    
    train_data = ImgPatches(
            root_dir = args.train_dir,
            patch_size = args.patch_size,
            stride = args.patch_stride,
            transform = transforms.CenterCrop(900),
    )
    
    train_data_noisy = ImgPatches(
            root_dir = args.train_dir,
            patch_size = args.patch_size,
            stride = args.patch_stride,
            transform = transforms.CenterCrop(900),
            patch_transform = AddRandomScatter(51,(15,20),(0.5,0.7),'uniform')
    )
    
    train_loader = DataLoader(
            ConcatDatasets(train_data[:1024],
                train_data_noisy[:1024]),
            batch_size=args.batch_size,
            shuffle = True,
    )
    
    validation_loader = DataLoader(
            ConcatDatasets(train_data[1024:1124], 
                train_data_noisy[1024:1124]),
            batch_size=args.batch_size,
            shuffle = True,
    )

    model = CNN().to(device)
    loss_fn = L2_SSIMLoss(a=0.8)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimiser, 'min')
    
    for t in range(args.epochs):
        print(f'Epoch {t+1}\n-------------------------------')
        lr = optimiser.param_groups[0]['lr']
        print(f'learning rate: {lr}')
        train(train_loader, model, loss_fn, optimiser)
        val_loss = validate(validation_loader, model, loss_fn)
        print(f'validation loss: {val_loss:>7f}\n')
        scheduler.step(val_loss)
    print('Done!')
    
    torch.save(model.state_dict(), 'trained.pth')

if __name__ == '__main__':
    main()
