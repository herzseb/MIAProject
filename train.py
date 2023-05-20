import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
import numpy as np
import wandb
from data import SegmentationDataset, collate_fn
from torch.utils.data import DataLoader, Subset
from aux import soft_dice_loss, soft_dice_score, FocalLoss, SoftTunedDiceBCELoss, hausdorff_distance
from net import UNet2D
import random

hyperparams = {'lr': 0.01, 'epochs': 200,  'criterion': 'CrossEntropy', 'batch_size': 8, 'accumulative_loss': 1, 'downsampling': 0.3, "conv_depths": (8,16,32,64)}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = SegmentationDataset("Dataset_BUSI_with_GT")
model = UNet2D(in_channels=1, out_channels=3, conv_depths=hyperparams.get("conv_depths")).to(device)
model.train()
dataloader = DataLoader(
            dataset, collate_fn=collate_fn, batch_size=hyperparams.get("batch_size"), shuffle=True)
optimizer = optim.SGD(model.parameters(), lr=hyperparams['lr'])
criterion = nn.CrossEntropyLoss()
for epoch in range(hyperparams['epochs']):
    total_loss = 0
    for i, item in enumerate(dataloader):
        if i > 50:
            break
        input, target = item
        # downsample images
        if hyperparams.get('downsampling') < 1:
            a = int(input.shape[2]*hyperparams.get('downsampling'))
            b = int(input.shape[3]*hyperparams.get('downsampling'))
            input = F.interpolate(input, (a,b))
            target = F.interpolate(target,  (a,b))
        input = input.to(device)
        target = target.cpu()
        target = torch.squeeze(target, dim=1)
        target = target.to(torch.long)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input)

        # Compute the loss
        loss = criterion(outputs.cpu(), target)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(total_loss / (i*hyperparams.get("batch_size")))
    
   