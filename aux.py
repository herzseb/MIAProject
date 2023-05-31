import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import cv2
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from skimage.draw import polygon
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def hausdorff_distance(pred, target):
    target = torch.squeeze(target, dim=0).cpu().numpy()
    pred = torch.squeeze(pred, dim=0).cpu().numpy()
    hausdorff_distance_1 = directed_hausdorff(target, pred)[0]
    hausdorff_distance_2 = directed_hausdorff(pred, target)[0]
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
    val = False
    y_true = F.one_hot(y_true, 3).permute(0,3,1,2)
    if len(y_pred.shape) < 4:
        val = True
        y_pred = F.one_hot(y_pred, 3).permute(0,3,1,2)
    intersection = torch.sum(y_pred * y_true, dim=(0, 2, 3))
    union = torch.sum(y_pred, dim=(0, 2, 3)) + torch.sum(y_true, dim=(0, 2, 3))
    dice = (2 * intersection + epsilon) / (union + epsilon)
    
    # Average the Dice coefficients across classes
    if val:
        mean_dice = torch.mean(dice)
    else:
        mean_dice = torch.sum(dice)/torch.count_nonzero(torch.sum(y_true, dim=(0, 2, 3)))
    return mean_dice

def soft_dice_loss(y_pred, y_true, epsilon=1e-6):
    return 1- soft_dice_score(y_pred, y_true, epsilon=1e-6)


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


# Framework taken from
# https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
# but adapted to pixel wise weighting
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, ignore_index=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets).view(-1)
        probs = torch.exp(-ce_loss)
        focal_loss= self.alpha * (1 - probs) ** self.gamma * ce_loss
        return torch.mean(focal_loss)


# 
def rectangle(points, num_points=20):
    p1 = points[0]
    p2 = points[1]
    p3 = points[2]
    p4 = points[3]
    x_interp_12 = np.linspace(p1[0], p2[0], num_points)[1:]
    y_interp_12 = np.linspace(p1[1], p2[1], num_points)[1:]
    interp_points_12 = np.column_stack((x_interp_12, y_interp_12))
    x_interp_23 = np.linspace(p2[0], p3[0], num_points)[1:]
    y_interp_23 = np.linspace(p2[1], p3[1], num_points)[1:]
    interp_points_23 = np.column_stack((x_interp_23, y_interp_23))
    x_interp_34 = np.linspace(p3[0], p4[0], num_points)[1:]
    y_interp_34 = np.linspace(p3[1], p4[1], num_points)[1:]
    interp_points_34 = np.column_stack((x_interp_34, y_interp_34))
    x_interp_41 = np.linspace(p4[0], p1[0], num_points)[1:]
    y_interp_41 = np.linspace(p4[1], p1[1], num_points)[1:]
    interp_points_41 = np.column_stack((x_interp_41, y_interp_41))
    interpolated_points = np.vstack((interp_points_12, interp_points_23, interp_points_34, interp_points_41))
    return interpolated_points


def snake_mask(mask):
    # threshold activations to binary mask
    _, mask = cv2.threshold(np.array(mask), 0.1, 1, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8) * 255
    # dilate to connect activations
    kernel_size = 5
    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask = cv2.dilate(mask, structuring_element)
    # get largest mask component
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    largest_component_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    mask = np.uint8(labels == largest_component_index) * 255
    # make inital rectangle that encloses all pixels of the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = np.vstack((contours))
    x, y, w, h = cv2.boundingRect(contours)
    rectangle_offset = 10
    x = x - rectangle_offset
    y = y - rectangle_offset
    w = w + rectangle_offset*2
    h = h + rectangle_offset*2
    rect = np.array([[y, x], [y + h, x], [y + h, x + w], [y, x + w], [y, x]])
    rect = rectangle(rect, num_points=50)
    # active contouring
    alpha = 0.15  # weight of polygon
    beta = 2.0  # weight of image gradient
    gamma = 0.001 # update setp
    iterations = 1000  # Number of iterations
    snake = active_contour(gaussian(mask), rect, alpha=alpha, beta=beta, gamma=gamma, max_num_iter=iterations)
    # add active contour to mask
    rr, cc = polygon(snake[:, 1], snake[:, 0])
    mask[cc-1, rr-1] = 255
    mask = cv2.erode(mask, structuring_element)
    mask = mask / 255
    outputs = torch.unsqueeze(torch.tensor(mask), dim=0).to(dtype=torch.long)
    outputs = outputs.to(device)
    return outputs
