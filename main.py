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
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)
# TODOS
# bring everything into memory

# Initialize wandb
wandb.init(project='MIA-project')

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Define your hyperparameter sets
hyperparameters = [
    {'lr': 0.01, 'epochs': 200,  'criterion': 'CrossEntropy', 'batch_size': 4, 'accumulative_loss': 1, 'downsampling': 1},
    {'lr': 0.001, 'epochs': 200, 'criterion': 'CrossEntropy', 'batch_size': 4, 'accumulative_loss': 1,  'downsampling': 1},
    {'lr': 0.0001, 'epochs': 200, 'criterion': 'CrossEntropy', 'batch_size': 4, 'accumulative_loss': 1,  'downsampling': 1}
]

wandb.log({"runs": hyperparameters})

# Define the number of folds for cross-validation
k_folds = 2

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

# pred = torch.tensor([[[[0.1,0.1,1],[0.1,0.1,1],[1,1,1]], [[0.9,0.9,0],[0.9,0.9,0],[0,0,0]], [[0,0,0],[0,0,0],[0,0,0]]]])
# target = torch.tensor([[[1,1,0],[1,1,0],[0,0,0]]])
# x = soft_dice_score(pred, target)

# Perform stratified k-fold cross-validation
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

for hyperparams in hyperparameters:
    fold_results_dice_mean = []
    fold_results_dice_std = []
    fold_results_HD_mean = []
    fold_results_HD_std = []
    fold_results = []
    param_loss = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(dataset, labels)):
        # Initialize the model and move it to the appropriate device
        model = UNet2D(in_channels=1, out_channels=3).to(device)

        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)

        # Define the loss function and optimizer
        if hyperparams.get("criterion") == "CrossEntropy":
            criterion = nn.CrossEntropyLoss()
        elif hyperparams.get("criterion") == "SoftDice":
            criterion = soft_dice_loss
        elif hyperparams.get("criterion") == "Focal":
            criterion = FocalLoss()
        elif hyperparams.get("criterion") == "SoftTunedDiceBCELoss":
            criterion = SoftTunedDiceBCELoss(
                total_epochs=hyperparams.get("epochs"))

        optimizer = optim.Adam(model.parameters(), lr=hyperparams['lr'])

        dataloader = DataLoader(
            train_dataset, collate_fn=collate_fn, batch_size=hyperparams.get("batch_size"), shuffle=True)

        model.train()
        fold_loss = []
        # Train the model
        for epoch in range(hyperparams['epochs']):
            epoch_loss = 0
            loss = 0
            print(f"epoch: {epoch}")
            for i, item in enumerate(dataloader):
                input, target = item
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
                outputs = F.softmax(model(input), dim=1).cpu()

                # Compute the loss
                if isinstance(criterion, SoftTunedDiceBCELoss):
                    acc_loss = criterion(outputs, target, epoch)
                else:
                    acc_loss = criterion(outputs, target)
                loss += acc_loss
                if (i+1)%hyperparams.get("batch_size") == 0:
                    loss = loss / hyperparams.get("batch_size")
                    # Backward pass and optimization
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    loss = 0
            fold_loss.append(epoch_loss/hyperparams['epochs'])
        param_loss.append(fold_loss)

        # Evaluate the model on the validation set
        model.eval()
        dice_normal = []
        dice_benign = []
        dice_malignant = []
        HD_normal = []
        HD_benign = []
        HD_malignant = []

        dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
        for i, item in enumerate(dataloader):
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
            outputs = torch.argmax(model(input), dim=1, keepdim=False)

            # Calculate the evaluation metric
            if torch.max(target) == 1:
                dice_benign.append(soft_dice_score(outputs, target).to("cpu"))
                target[target > 0] = 1
                outputs[outputs > 0] = 1
                HD_benign.append(hausdorff_distance(outputs, target))
            elif torch.max(target) == 2:
                dice_malignant.append(
                    soft_dice_score(outputs, target).to("cpu"))
                target[target > 0] = 1
                outputs[outputs > 0] = 1
                HD_malignant.append(hausdorff_distance(outputs, target))
            elif torch.max(target) == 0:
                dice_normal.append(soft_dice_score(outputs, target).to("cpu"))

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

        # Store the evaluation metric
        fold_results_dice_mean.append(
            [dice_benign_mean, dice_normal_mean, dice_malignant_mean])
        fold_results_dice_std.append(
            [dice_bening_std, dice_normal_std, dice_malignant_std])
        fold_results_HD_mean.append(
            [HD_benign_mean, HD_malignant_mean])
        fold_results_HD_std.append(
            [HD_bening_std, HD_malignant_std])

        # Log the evaluation metric for the current fold
        # wandb.log(
        # #     {f'Fold dice mean {fold + 1} {"fold_results"}': fold_results})

        # table_dice_mean = wandb.Table(data=fold_results_dice_mean, columns=["normal", "benign", "malignant"])
        # table_dice_std = wandb.Table(data=fold_results_dice_std, columns=["normal", "benign", "malignant"])
        # table_HD_mean = wandb.Table(data=fold_results_HD_mean, columns=["normal", "benign", "malignant"])
        # table_HD_std = wandb.Table(data=fold_results_HD_std, columns=["normal", "benign", "malignant"])

        # wandb.log({f"Fold {fold + 1} Dice mean": wandb.plot.line(table_dice_mean, "normal", "benign", "malignant", title=f"Fold {fold + 1} Dice mean")})
        # wandb.log({f"Fold {fold + 1} Dice std": wandb.plot.line(table_dice_std, "normal", "benign", "malignant", title=f"Fold {fold + 1} Dice std")})
        # wandb.log({f"Fold {fold + 1} HD mean": wandb.plot.line(table_HD_mean, "normal", "benign", "malignant", title=f"Fold {fold + 1} HD mean")})
        # wandb.log({f"Fold {fold + 1} HD std": wandb.plot.line(table_HD_std, "normal", "benign", "malignant", title=f"Fold {fold + 1} HD std")})

    paramset_dice_benign_mean = np.mean(fold_results_dice_mean, axis=0)[0]
    paramset_dice_normal_mean = np.mean(fold_results_dice_mean, axis=0)[1]
    paramset_dice_malignant_mean = np.mean(fold_results_dice_mean, axis=0)[2]
    paramset_dice_bening_std = np.std(fold_results_dice_std, axis=0)[0]
    paramset_dice_normal_std = np.std(fold_results_dice_std, axis=0)[1]
    paramset_dice_malignant_std = np.std(fold_results_dice_std, axis=0)[2]
    paramset_HD_benign_mean = np.mean(fold_results_HD_mean, axis=0)[0]
    paramset_HD_malignant_mean = np.mean(fold_results_HD_mean, axis=0)[1]
    paramset_HD_bening_std = np.std(fold_results_HD_std, axis=0)[0]
    paramset_HD_malignant_std = np.std(fold_results_HD_std, axis=0)[1]

    # Calculate mean and standard deviation for the current hyperparameter set
    paramset_dice_mean = np.mean(fold_results_dice_mean)
    paramset_dice_std = np.std(fold_results_dice_std)
    paramset_HD_mean = np.mean(fold_results_HD_mean)
    paramset_HD_std = np.std(fold_results_HD_std)

    # Log the mean and standard deviation for the current hyperparameter set
    wandb.log({f'Dice mean': paramset_dice_mean,
              f'Dice std': paramset_dice_std,
               f'HD mean': paramset_HD_mean,
               f'HD std': paramset_HD_std,
               f'Dice bening mean': paramset_dice_benign_mean,
               f'Dice normal mean': paramset_dice_normal_mean,
               f'Dice malignant mean': paramset_dice_malignant_mean,
               f'Dice bening std': paramset_dice_bening_std,
               f'Dice normal std': paramset_dice_normal_std,
               f'Dice malignant std': paramset_dice_malignant_std,
               f'HD bening mean': paramset_HD_benign_mean,
               f'HD malignant mean': paramset_HD_malignant_mean,
               f'HD bening std': paramset_HD_bening_std,
               f'HD malignant std': paramset_HD_malignant_std})

    param_loss = np.mean(param_loss, axis=0).tolist()
    param_loss_with_epoch = [[item, i] for i, item in enumerate(param_loss)]
    table_param_loss = wandb.Table(
        data=param_loss_with_epoch, columns=["loss", "epoch"])
    wandb.log({f"{hyperparams.get('criterion')}": wandb.plot.line(
        table_param_loss, "loss", "epoch", title=f"Average loss per epoch over {k_folds} folds")})

    # Print the results
    print('Hyperparameters:', hyperparams)
    print('Dice Mean:', paramset_dice_mean)
    print('Dice Standard Deviation', paramset_dice_std)
    print('HD Mean:', paramset_HD_mean)
    print('HD Standard Deviation', paramset_HD_std)
    print('Loss', param_loss)
