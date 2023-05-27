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
import matplotlib.pyplot as plt
hyperparams = {'lr': 0.01, 'epochs': 20000,  'criterion': 'CrossEntropy', 'batch_size': 2, 'accumulative_loss': 1, 'downsampling': 0.3, "conv_depths": (8,16,32,64)}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = SegmentationDataset("Dataset_BUSI_with_GT")
model = UNet2D(in_channels=1, out_channels=3, conv_depths=hyperparams.get("conv_depths")).to(device)
model.train()
dataloader = DataLoader(
            dataset, collate_fn=collate_fn, batch_size=hyperparams.get("batch_size"), shuffle=True)
# optimizer = optim.SGD(model.parameters(), lr=hyperparams['lr'], momentum=0.95, weight_decay=0.001)
optimizer = optim.Adam(model.parameters(), lr=hyperparams['lr'])
criterion = soft_dice_loss
x = torch.randn(1, 1, 224, 224)
y = torch.randint(0, 2, (1, 1, 224, 224)).float()
for epoch in range(hyperparams['epochs']):
    total_loss = 0
    for i, item in enumerate(dataloader):
        if i > 10:
            break
        #input, target = item
        input, target = dataset[200]
        input = torch.unsqueeze(input,dim=0)
        target = torch.unsqueeze(target,dim=0)
        # downsample images
        if hyperparams.get('downsampling') < 1:
            a = int(input.shape[2]*hyperparams.get('downsampling'))
            b = int(input.shape[3]*hyperparams.get('downsampling'))
            input = F.interpolate(input, (a,b))
            target = F.interpolate(target,  (a,b))
        
        input = input.to(device)
        target = target.to(device)
        target = torch.squeeze(target, dim=1)
        target = target.to(torch.long)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input)
        outputs = F.softmax(outputs, dim=1)

        # Compute the loss
        loss = criterion(outputs, target)

        print("val ", soft_dice_score(torch.argmax(outputs, dim=1, keepdim=False), target))
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(total_loss / (i+1))
    
   