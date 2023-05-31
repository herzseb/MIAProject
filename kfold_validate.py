
"""
evaluate model with postprocessing on validation folds
"""
import cv2
from threading import active_count
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
import wandb
from data import SegmentationDataset, collate_fn
from torch.utils.data import DataLoader, Subset
from aux import soft_dice_loss, soft_dice_score, FocalLoss, SoftTunedDiceBCELoss, hausdorff_distance, snake_mask
from net import UNet2D
import random
import matplotlib.pyplot as plt
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


# Initialize wandb
wandb.init(project='MIA-project')
hyperparams = [{'modelpath': "Unet_20230531-131629.pt",  'downsampling': 0.5, "conv_depths": (32, 64, 128, 256, 512), 'postprocess': False},
               {'modelpath': "Unet_20230531-135323.pt",  'downsampling': 0.5,"conv_depths": (32, 64, 128, 256, 512), 'postprocess': False},
               {'modelpath': "Unet_20230531-142958.pt",  'downsampling': 0.5, "conv_depths": (32, 64, 128, 256, 512), 'postprocess': False}]
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

# generate indices: instead of the actual data we pass in integers instead
train_indices, test_indices, train_labels, test_labels = train_test_split(range(len(
    dataset)), dataset.labels, stratify=dataset.labels, test_size=0.15, random_state=42)
train_set = Subset(dataset, train_indices)
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(train_set, train_labels)):
    val_dataset = Subset(train_set, val_idx)

    # generate subset based on indices
    test_set = Subset(dataset, test_indices)
    # model = torch.load(hyperparams["modelpath"])
    model = UNet2D(in_channels=1, out_channels=3,
                   conv_depths=hyperparams[fold].get("conv_depths"))
    model.load_state_dict(torch.load(hyperparams[fold]["modelpath"]))
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
    dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    for i, item in enumerate(dataloader):
        with torch.no_grad():
            input, target = item
            if hyperparams[fold].get('downsampling') < 1:
                a = int(input.shape[2]*hyperparams[fold].get('downsampling'))
                b = int(input.shape[3]*hyperparams[fold].get('downsampling'))
                input = F.interpolate(input, (a, b))
                target = F.interpolate(target,  (a, b))
            input = input.to(device)
            target = target.to(device)
            target = torch.squeeze(target, dim=1)
            target = target.to(torch.long)

            # Forward pass
            outputs = model(input)
            if hyperparams[fold]["postprocess"]:
                # convert to likelihoods
                probs = F.softmax(outputs, dim=1)
                probs = probs.cpu()
                # get most common class, if non-background class, apply postprocessing
                if torch.sum(1-probs[0, 0]) < 50:
                    outputs = torch.argmax(outputs, dim=1, keepdim=False)
                else:
                    if torch.sum(probs[0, 1]) > torch.sum(probs[0, 2]):
                        mask = probs[0, 1]
                        outputs = snake_mask(mask)
                    else:
                        mask = probs[0, 2]
                        outputs = snake_mask(mask)
                        outputs = outputs * 2
            else:
                outputs = torch.argmax(outputs, dim=1)

            # Calculate the evaluation metric, hausdorf requires binary mask
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

            # # extract contours for plotting
            # contours_target, _ = cv2.findContours(target[0].cpu().numpy().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # contours_mask, _ = cv2.findContours(outputs[0].cpu().numpy().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # plt.imshow(input[0,0].cpu())
            # if contours_target:
            #     contours_target = np.vstack((contours_target[0], np.expand_dims(contours_target[0][0], axis=0)))
            #     plt.plot(contours_target[:,0,0], contours_target[:,0,1], 'g')
            # if contours_mask:
            #     contours_mask = np.vstack((contours_mask[0], np.expand_dims(contours_mask[0][0], axis=0)))
            #     plt.plot(contours_mask[:,0,0], contours_mask[:,0,1], 'r')
            # plt.savefig(f"out/{i}.png")
            # plt.cla()


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
    print('Hyperparameters: ', hyperparams[fold])
    print('Dice bening mean: ', dice_benign_mean)
    print('Dice normal mean: ', dice_normal_mean)
    print('Dice malignant mean: ', dice_malignant_mean)
    print('Dice Mean: ', np.mean(
        [dice_benign_mean, dice_normal_mean, dice_malignant_mean]))
    print('Dice Standard Deviation: ', np.mean(
        [dice_bening_std, dice_normal_std, dice_malignant_std]))
    print('HD Mean: ',  np.mean([HD_benign_mean, HD_malignant_mean]))
    print('HD Standard Deviation: ',  np.mean([HD_bening_std, HD_malignant_std]))
