from copy import deepcopy
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
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)
import time

debugging = False
# TODOS
# bring everything into memory 

# Initialize wandb
wandb.init(project='MIA-project')

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Define your hyperparameter sets
hyperparameters = [
    {'lr': 0.001, 'epochs': 250,  'criterion': 'SoftDice', 'batch_size': 12, 'acc_loss': 1, 'downsample': 0.5, "features": (32, 64, 128, 256, 512), "dp": 0.5},
]

wandb.log({"runs": hyperparameters})

# Define the number of folds for cross-validation
k_folds = 2

# Define the dataset and labels (assuming binary classification)
dataset = SegmentationDataset("Dataset_BUSI_with_GT", augment=True)

print(
    f"dataset: benign: {dataset.labels.count('benign')}, malignant: {dataset.labels.count('malignant')}, normal: {dataset.labels.count('normal')}")

# generate indices: instead of the actual data we pass in integers instead
train_indices, test_indices, train_labels, test_labels = train_test_split(range(len(dataset)), dataset.labels, stratify=dataset.labels, test_size=0.15, random_state=42)

# generate subset based on indices
train_set = Subset(dataset, train_indices)
test_set = Subset(dataset, test_indices)
# Test dice score manually
# pred = torch.tensor([[[[0.1,0.1,1],[0.1,0.1,1],[1,1,1]], [[0.9,0.9,0],[0.9,0.9,0],[0,0,0]], [[0,0,0],[0,0,0],[0,0,0]]]])
# target = torch.tensor([[[1,1,0],[1,1,0],[0,0,0]]])
# x = soft_dice_score(pred, target)

# Perform stratified k-fold cross-validation
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
# loop over all hyperparameters
for hyperparams in hyperparameters:
    fold_results_dice_mean = []
    fold_results_dice_std = []
    fold_results_HD_mean = []
    fold_results_HD_std = []
    fold_results = []
    param_train_loss = []
    val_loss = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_set, train_labels)):
        # for greedy optimization act as normal train validation split
        if k_folds == 2:
            if fold == 1:
                break
        # Initialize the model and move it to the appropriate device
        model = UNet2D(in_channels=1, out_channels=3, conv_depths=hyperparams.get("features"), dropout=hyperparams.get("dp")).to(device)
        
        # creat train and validation set for current split
        train_dataset = Subset(train_set, train_idx)
        val_dataset = Subset(train_set, val_idx)

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

        optimizer = optim.SGD(model.parameters(), lr=hyperparams['lr'], momentum=0.95, weight_decay=0.0001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

        # dataloader with custom batch function such that all images in a batch get padded to the largest one
        dataloader = DataLoader(
            train_dataset, collate_fn=collate_fn, batch_size=hyperparams.get("batch_size"), shuffle=True)

        
        fold_train_loss = []
        fold_val_loss = []
        # Train the model
        for epoch in range(hyperparams['epochs']):
            model.train()
            epoch_loss = 0
            for i, item in enumerate(dataloader):
                input, target = item
                # for debugging only take a small portion of the dataset
                if debugging:
                    if i > 10:
                        break
                # downsample images
                if hyperparams.get('downsample') < 1:
                    a = int(input.shape[2]*hyperparams.get('downsample'))
                    b = int(input.shape[3]*hyperparams.get('downsample'))
                    input = F.interpolate(input, (a,b))
                    target = F.interpolate(target,  (a,b))
                input = input.to(device)
                target = target.cpu()
                target = torch.squeeze(target, dim=1)
                target = target.to(torch.long)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(input).cpu()

                # Compute the loss
                if isinstance(criterion, SoftTunedDiceBCELoss):
                    outputs = F.softmax(outputs, dim=1)
                    loss = criterion(outputs, target, epoch)
                elif criterion == soft_dice_loss:
                    outputs = F.softmax(outputs, dim=1)
                    loss = criterion(outputs, target)
                else:
                    loss = criterion(outputs, target)
                
                # accumulated loss to simulate larger batch sizes
                if (i+1)%hyperparams.get("acc_loss") == 0:
                    loss = loss / hyperparams.get("acc_loss")
                    # Backward pass and optimization
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
            fold_train_loss.append(epoch_loss/(i+1))
            print(f"epoch: {epoch}, loss: {epoch_loss/(i+1)}")
        

            # Evaluate the model on the validation set
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
                    if hyperparams.get('downsample') < 1:
                        a = int(input.shape[2]*hyperparams.get('downsample'))
                        b = int(input.shape[3]*hyperparams.get('downsample'))
                        input = F.interpolate(input, (a,b))
                        target = F.interpolate(target,  (a,b))
                    input = input.to(device)
                    target = target.to(device)
                    target = torch.squeeze(target, dim=1)
                    target = target.to(torch.long)

                    # Forward pass
                    outputs = model(input)
                    outputs = torch.argmax(outputs, dim=1, keepdim=False)
                    
                    # if isinstance(criterion, SoftTunedDiceBCELoss):
                    #     outputs = torch.argmax(outputs, dim=1, keepdim=False)
                    #     loss += criterion(outputs, target, epoch)
                    # elif isinstance(criterion, nn.CrossEntropyLoss):
                    #     loss += criterion(outputs, target)
                    #     outputs = torch.argmax(outputs, dim=1, keepdim=False)
                    # else:
                    #     outputs = torch.argmax(outputs, dim=1, keepdim=False)
                    #     loss += criterion(outputs, target)
                    # evaluate only on dice score since its teh score we are optimizing for
                    loss += soft_dice_loss(outputs, target)
                # evaluate all metrics only in last epoch aka when the model is trained the best
                if epoch == hyperparams['epochs']-1:
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

            loss = loss.item()
            fold_val_loss.append(loss/(i+1))
            if loss/(i+1) <= np.min(fold_val_loss):
                print("updated best param set")
                best_params = deepcopy(model.state_dict())
            
        
            scheduler.step()

        val_loss.append(fold_val_loss)

        param_train_loss.append(fold_train_loss)

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

    param_train_loss = np.mean(param_train_loss, axis=0).tolist()
    val_loss = np.mean(val_loss, axis=0).tolist()
    param_train_loss_with_epoch = [[train, i] for i, train in enumerate(param_train_loss)]
    param_val_loss_with_epoch = [[val, i] for i, val in enumerate(val_loss)]
    if k_folds == 2:
        print_folds = 1
    table_param_train_loss = wandb.Table(
        data=param_train_loss_with_epoch, columns=["training loss", "epoch"])
    wandb.log({f"{hyperparams.get('criterion')} training loss {hyperparams}": wandb.plot.line(
        table_param_train_loss, "epoch","training loss", title=f"Average {hyperparams.get('criterion')} loss per epoch over {print_folds} folds")})
    table_val_loss_with_epoch = wandb.Table(
        data=param_val_loss_with_epoch, columns=["validation loss", "epoch"])
    wandb.log({f"{hyperparams.get('criterion')} validation loss {hyperparams}": wandb.plot.line(
        table_val_loss_with_epoch, "epoch", "validation loss", title=f"Average {hyperparams.get('criterion')} loss per epoch over {print_folds} folds")})

    # Print the results
    print('Hyperparameters:', hyperparams)
    print('Dice Mean:', paramset_dice_mean)
    print('Dice Standard Deviation', paramset_dice_std)
    print('HD Mean:', paramset_HD_mean)
    print('HD Standard Deviation', paramset_HD_std)
    print('Train Loss', param_train_loss)
    print('Validation Loss', val_loss)
    path = "Unet_"
    path += time.strftime("%Y%m%d-%H%M%S")
    path += ".pt"
    print(f'Save state dict to {path}')
    torch.save(best_params, path)
