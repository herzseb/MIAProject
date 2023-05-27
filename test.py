import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
import wandb
from data import SegmentationDataset, collate_fn
from torch.utils.data import DataLoader, Subset
from aux import soft_dice_loss, soft_dice_score, FocalLoss, SoftTunedDiceBCELoss, hausdorff_distance
from net import UNet2D
import random
import matplotlib.pyplot as plt
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Initialize wandb
wandb.init(project='MIA-project')

hyperparams = {'modelpath':"Unet_20230527-115530.pt",  'downsampling': 0.5, "conv_depths": (32, 64, 128, 256, 512)}
wandb.log({"params": hyperparams})
# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Define the dataset and labels (assuming binary classification)
dataset = SegmentationDataset("Dataset_BUSI_with_GT")

# dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
labels = []
for path in dataset.file_list:
    if "benign" in path[0]:
        labels.append("benign")
    elif "malignant" in path[0]:
        labels.append("malignant")
    elif "normal" in path[0]:
        labels.append("normal")
print(
    f"dataset: benign: {labels.count('benign')}, malignant: {labels.count('malignant')}, normal: {labels.count('normal')}")

# generate indices: instead of the actual data we pass in integers instead
train_indices, test_indices, _, _ = train_test_split(range(len(dataset)), labels, stratify=labels, test_size=0.15, random_state=42)

# generate subset based on indices
train_set = Subset(dataset, train_indices)
test_set = Subset(dataset, test_indices)
model = torch.load(hyperparams["modelpath"])
# model = UNet2D(in_channels=1, out_channels=3, conv_depths=hyperparams.get("conv_depths"))
# model.load_state_dict(torch.load(hyperparams["modelpath"]))
model.eval()
model = model.to(device)
# Evaluate the model on the test set
model.eval()
dice_normal = []
dice_benign = []
dice_malignant = []
HD_normal = []
HD_benign = []
HD_malignant = []
loss = 0
# use singel batch size fo easier data handeling
# evaluate at every epoch for early stopping
dataloader = DataLoader(test_set, batch_size=1, shuffle=True)
fig, ax = plt.subplots(3)
for i, item in enumerate(dataloader):
    with torch.no_grad():
        input, target = item
        if hyperparams.get('downsampling') < 1:
            a = int(input.shape[2]*hyperparams.get('downsampling'))
            b = int(input.shape[3]*hyperparams.get('downsampling'))
            input = F.interpolate(input, (a,b))
            target = F.interpolate(target,  (a,b))
        input = input.to(device)
        target = target.to(device)
        target = torch.squeeze(target, dim=1)
        target = target.to(torch.long)

        # Forward pass
        outputs = model(input)
        outputs = torch.argmax(outputs, dim=1, keepdim=False)
        
    
        # Calculate the evaluation metric
        if torch.max(target) == 1:
            dice_benign.append(soft_dice_score(outputs, target).to("cpu"))
            outputs[outputs != 1] = 0
            HD_benign.append(hausdorff_distance(outputs, target))
        elif torch.max(target) == 2:
            dice_malignant.append(
                soft_dice_score(outputs, target).to("cpu"))
            outputs[outputs != 2] = 0
            outputs[outputs > 0] = 1
            target[target > 0] = 1
            HD_malignant.append(hausdorff_distance(outputs, target))
        elif torch.max(target) == 0:
            dice_normal.append(soft_dice_score(outputs, target).to("cpu"))

        ax[0].imshow(input[0,0].cpu())
        ax[1].imshow(target[0].cpu())
        ax[2].imshow(outputs[0].cpu())
        plt.savefig(f"out/{i}.png")

dice_benign_mean = np.mean(dice_benign)
dice_bening_std = np.std(dice_benign)
dice_normal_mean = np.mean(dice_normal)
dice_normal_std = np.std(dice_normal)
dice_malignant_mean = np.mean(dice_malignant)
dice_malignant_std = np.std(dice_malignant)
HD_benign_mean = np.mean(HD_benign)
HD_bening_std = np.std(HD_benign)
HD_malignant_mean = np.mean(HD_malignant)
HD_malignant_std = np.std(HD_malignant)

# Log the mean and standard deviation for the current hyperparameter set
wandb.log({f'Dice mean': np.mean([dice_benign_mean, dice_normal_mean, dice_malignant_mean]),
            f'Dice std': np.mean([dice_bening_std, dice_normal_std, dice_malignant_std]),
            f'HD mean': np.mean([HD_benign_mean, HD_malignant_mean]),
            f'HD std': np.mean([HD_bening_std, HD_malignant_std]),
            f'Dice bening mean': dice_benign_mean,
            f'Dice normal mean': dice_normal_mean,
            f'Dice malignant mean': dice_malignant_mean,
            f'Dice bening std': dice_bening_std,
            f'Dice normal std': dice_normal_std,
            f'Dice malignant std': dice_malignant_std,
            f'HD bening mean': HD_benign_mean,
            f'HD malignant mean': HD_malignant_mean,
            f'HD bening std': HD_bening_std,
            f'HD malignant std': HD_malignant_std})


# Print the results
print('Hyperparameters: ', hyperparams)
print('Dice Mean: ', np.mean([dice_benign_mean, dice_normal_mean, dice_malignant_mean]))
print('Dice Standard Deviation: ', np.mean([dice_bening_std, dice_normal_std, dice_malignant_std]))
print('HD Mean: ',  np.mean([HD_benign_mean, HD_malignant_mean]))
print('HD Standard Deviation: ',  np.mean([HD_bening_std, HD_malignant_std]))