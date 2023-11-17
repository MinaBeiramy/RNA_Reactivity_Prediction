import torch
import torch.nn.functional as F

from crnn import CRNN
import numpy as np

import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torch.nn import CTCLoss

from dataloaders import CNNDataset
from config import *


# 📉 Define loss functions for training and evaluation
def loss_fn(output, target):
    # 🪟 Clip the target values to be within the range [0, 1]
    clipped_target = torch.clip(target, min=0, max=1)
    # 📉 Calculate the mean squared error loss
    mses = F.mse_loss(output, clipped_target, reduction='mean')
    return mses

def mae_fn(output, target):
    # 🪟 Clip the target values to be within the range [0, 1]
    clipped_target = torch.clip(target, min=0, max=1)
    # 📉 Calculate the mean absolute error loss
    maes = F.l1_loss(output, clipped_target, reduction='mean')
    return maes

def train_batch(crnn, data, optimizer, criterion, device):
    crnn.train()
    images, targets, target_lengths = [d.to(device) for d in data]
    print(images.shape())
    logits = crnn(images)
    #log_probs = torch.nn.functional.log_softmax(logits, dim=2)
    log_probs = logits

    batch_size = images.size(0)
    input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
    target_lengths = torch.flatten(target_lengths)

    loss = criterion(log_probs, targets)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(crnn.parameters(), 5) # gradient clipping with 5
    optimizer.step()
    return loss.item()


def main():
    epochs = 1
    dataset = CNNDataset(data_csv=P_TRAIN_CSV, target_csv=P_TARGETS_CSV, matrix_dir=ETERNA_PKG_BPP )
    generator1 = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [0.7, 0.3], generator1)
    dataset = CNNDataset(data_csv=P_TRAIN_CSV, target_csv=P_TARGETS_CSV, matrix_dir=ETERNA_PKG_BPP )
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, persistent_workers=True)  
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, persistent_workers=True) 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    model = CRNN(1, 224, 224, 1)
    #if reload_checkpoint:
    #    crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device))
    model.to(device)

    # 📈 Define the optimizer with learning rate and weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=5e-4)

    # 🚂 Iterate over epochs
    for epoch in range(epochs):
        train_losses = []
        train_maes = []
        model.train()
        
        # 🚞 Iterate over batches in the training dataloader
        for batch in (pbar := tqdm(train_dataloader, position=0, leave=True)):
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            out = torch.squeeze(out)
            loss = loss_fn(out[batch.valid_mask], batch.y[batch.valid_mask])
            mae = mae_fn(out[batch.valid_mask], batch.y[batch.valid_mask])
            loss.backward()
            train_losses.append(loss.detach().cpu().numpy())
            train_maes.append(mae.detach().cpu().numpy())
            optimizer.step()
            pbar.set_description(f"Train loss {loss.detach().cpu().numpy():.4f}")
        
        # 📊 Print average training loss and MAE for the epoch
        print(f"Epoch {epoch} train loss: ", np.mean(train_losses))
        print(f"Epoch {epoch} train mae: ", np.mean(train_maes))
        
        val_losses = []
        val_maes = []
        model.eval()
        
        # 🚞 Iterate over batches in the validation dataloader
        for batch in (pbar := tqdm(val_dataloader, position=0, leave=True)):
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            out = torch.squeeze(out)
            loss = loss_fn(out[batch.valid_mask], batch.y[batch.valid_mask])
            mae = mae_fn(out[batch.valid_mask], batch.y[batch.valid_mask])
            val_losses.append(loss.detach().cpu().numpy())
            val_maes.append(mae.detach().cpu().numpy())
            pbar.set_description(f"Validation loss {loss.detach().cpu().numpy():.4f}")
        
        # 📊 Print average validation loss and MAE for the epoch
        print(f"Epoch {epoch} val loss: ", np.mean(val_losses))
        print(f"Epoch {epoch} val mae: ", np.mean(val_maes))

if __name__ == '__main__':
    main()

