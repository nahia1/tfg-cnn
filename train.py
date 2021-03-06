import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms

from dataset import ImgPatches, ConcatDatasets, LazyDataset
from loss import L2_SSIMLoss
from transforms import AddRandomScatter
from models import CNN

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('train_dir', type=str,
        help='directory containing the training dataset')
parser.add_argument('--epochs', type=int, default=10, 
        help='number of training epochs (default 10)')
parser.add_argument('--learning-rate', type=float, default=1e-3, 
        help='initial learning rate (default 1e-3)')
parser.add_argument('--batch-size', type=int, default=64,
        help='size of the mini-batches (default 64)')
parser.add_argument('--patch-size', type=int, default=64,
        help='size of the sliding window for patch extraction (default 64)')
parser.add_argument('--patch-stride', type=int, default=64,
        help='stride of the sliding window for patch extraction (default 64)')
parser.add_argument('--val-split', type=float, default=0.8,
        help='validation split (default 0.8)')

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
        torch.cuda.empty_cache()
        
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

    data = ImgPatches(
            root_dir = args.train_dir,
            patch_size = args.patch_size,
            stride = args.patch_stride,
    )
    
    data_noisy = ImgPatches(
            root_dir = args.train_dir,
            patch_size = args.patch_size,
            stride = args.patch_stride,
            patch_transform = AddRandomScatter(71,(15,25),(0.4,0.8),'uniform')
    )

    #data = LazyDataset(
    #        root_dir = args.train_dir,
    #)
    #
    #data_noisy = LazyDataset(
    #        root_dir = args.train_dir,
    #        transform = AddRandomScatter(101,(25,35),(0.4,0.8),'uniform')
    #)
    
    n_train = int(args.val_split*len(data))
    n_val = len(data) - n_train

    train_data, validation_data = torch.utils.data.random_split(
            ConcatDatasets(data, data_noisy),
            [n_train, n_val]
    )

    train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            num_workers = 12,
    )
    
    validation_loader = DataLoader(
            validation_data,
            batch_size=args.batch_size,
            num_workers = 12,
    )

    model = CNN().to(device)
    if torch.cuda.device_count() > 1:
       print('Using', torch.cuda.device_count(), 'GPUs')
       model = nn.DataParallel(model)

    loss_fn = L2_SSIMLoss(a=0.2).to(device)
    optimiser = torch.optim.Adam(model.parameters(), 
            lr=args.learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimiser, 'min', patience = 5, min_lr=1e-5)
    
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
