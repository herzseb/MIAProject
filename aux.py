import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import directed_hausdorff

def hausdorff_distance(pred, target):
    # Convert the masks to numpy arrays if they are not already
    target = torch.squeeze(target, dim=0).cpu().numpy()
    pred = torch.squeeze(pred, dim=0).cpu().numpy()
    
    # Calculate the directed Hausdorff distance from ground truth to prediction
    hausdorff_distance_1 = directed_hausdorff(target, pred)[0]
    
    # Calculate the directed Hausdorff distance from prediction to ground truth
    hausdorff_distance_2 = directed_hausdorff(pred, target)[0]
    
    # Take the maximum of the two directed distances
    hausdorff_distance = max(hausdorff_distance_1, hausdorff_distance_2)
    
    return hausdorff_distance



def soft_dice_score(y_pred, y_true, epsilon=1e-6): 
    ''' 
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
  
    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax) 
        epsilon: Used for numerical stability to avoid divide by zero errors
    
    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation 
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation 
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)
        
        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    '''
    
    # skip the batch and class axis for calculating Dice score
    #axes = (2,3)
    y_true = F.one_hot(y_true, 3).permute(0,3,1,2)
    if len(y_pred.shape) < 4:
        y_pred = F.one_hot(y_pred, 3).permute(0,3,1,2)
    # numerator = 2. * torch.sum(y_pred * y_true, axes)
    # denominator = torch.sum(torch.square(y_pred) + torch.square(y_true), axes)
    intersection = torch.sum(y_pred * y_true, dim=(0, 2, 3))
    union = torch.sum(y_pred, dim=(0, 2, 3)) + torch.sum(y_true, dim=(0, 2, 3))
    
    # Compute Dice coefficient for each class
    dice = (2 * intersection + epsilon) / (union + epsilon)
    
    # Average the Dice coefficients across classes
    mean_dice = torch.mean(dice)
    return mean_dice
    # return torch.mean((numerator + epsilon) / (denominator + epsilon)) # average over classes and batch

def soft_dice_loss(y_true, y_pred, epsilon=1e-6):
    return 1- soft_dice_score(y_true, y_pred, epsilon=1e-6)



# https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
class SoftTunedDiceBCELoss(nn.Module):
    def __init__(self, total_epochs):
        super(SoftTunedDiceBCELoss, self).__init__()
        self.total_epochs = total_epochs

    def forward(self, inputs, targets, epoch):       
        
        #flatten label and prediction tensors
        dice_loss = soft_dice_loss(inputs, targets)
        targets = targets.to(torch.long)
        CE = F.cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = ((self.total_epochs - epoch)/ self.total_epochs) * CE + (epoch / self.total_epochs) * dice_loss
        
        return Dice_BCE



# https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = 0.8
        self.gamma = 2

    def forward(self, inputs, targets, smooth=1):      
        CE = F.cross_entropy(inputs, targets, reduction='mean')
        CE_EXP = torch.exp(-CE)
        focal_loss = self.alpha * (1-CE_EXP)**self.gamma * CE
                       
        return focal_loss


