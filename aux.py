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
    val = False
    y_true = F.one_hot(y_true, 3).permute(0,3,1,2)
    if len(y_pred.shape) < 4:
        val = True
        y_pred = F.one_hot(y_pred, 3).permute(0,3,1,2)
    # numerator = 2. * torch.sum(y_pred * y_true, axes)
    # denominator = torch.sum(torch.square(y_pred) + torch.square(y_true), axes)
    intersection = torch.sum(y_pred * y_true, dim=(0, 2, 3))
    union = torch.sum(y_pred, dim=(0, 2, 3)) + torch.sum(y_true, dim=(0, 2, 3))
    
    # Compute Dice coefficient for each class
    dice = (2 * intersection + epsilon) / (union + epsilon)
    
    # Average the Dice coefficients across classes
    if val:
        mean_dice = torch.mean(dice)
    else:
        mean_dice = torch.sum(dice)/torch.count_nonzero(torch.sum(y_true, dim=(0, 2, 3)))
    return mean_dice
    # return torch.mean((numerator + epsilon) / (denominator + epsilon)) # average over classes and batch

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



# https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
# class FocalLoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.alpha = 0.8
#         self.gamma = 2

#     def forward(self, inputs, targets, smooth=1):      
#         CE = F.cross_entropy(inputs, targets, reduction='mean')
#         CE_EXP = torch.exp(-CE)
#         focal_loss = self.alpha * (1-CE_EXP)**self.gamma * CE
                       
#         return focal_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, ignore_index=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets).view(-1)

        if self.ignore_index is not None:
            valid_indices = targets != self.ignore_index
            inputs = inputs[valid_indices]
            targets = targets[valid_indices]

        probs = torch.exp(-ce_loss)
        focal_loss= self.alpha * (1 - probs) ** self.gamma * ce_loss
        return torch.mean(focal_loss)


def rectangle(points, num_points=20):
    p1 = points[0]
    p2 = points[1]
    p3 = points[2]
    p4 = points[3]

    x_interp_12 = np.linspace(p1[0], p2[0], num_points+2)[1:-1]
    y_interp_12 = np.linspace(p1[1], p2[1], num_points+2)[1:-1]
    interp_points_12 = np.column_stack((x_interp_12, y_interp_12))
    x_interp_23 = np.linspace(p2[0], p3[0], num_points+2)[1:-1]
    y_interp_23 = np.linspace(p2[1], p3[1], num_points+2)[1:-1]
    interp_points_23 = np.column_stack((x_interp_23, y_interp_23))
    x_interp_34 = np.linspace(p3[0], p4[0], num_points+2)[1:-1]
    y_interp_34 = np.linspace(p3[1], p4[1], num_points+2)[1:-1]
    interp_points_34 = np.column_stack((x_interp_34, y_interp_34))
    x_interp_41 = np.linspace(p4[0], p1[0], num_points+2)[1:-1]
    y_interp_41 = np.linspace(p4[1], p1[1], num_points+2)[1:-1]
    interp_points_41 = np.column_stack((x_interp_41, y_interp_41))
    interpolated_points = np.vstack((interp_points_12, interp_points_23, interp_points_34, interp_points_41))
    return interpolated_points

def snake_mask(mask):
# Step 1: Extract the edges from the segmentation mask
    _, mask = cv2.threshold(np.array(mask), 0.1, 1, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8) * 255
    kernel_size = 5
    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # Perform morphological dilation to grow the regions
    mask = cv2.dilate(mask, structuring_element)

            # Perform connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    # Find the index of the largest component (excluding background label)
    largest_component_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

    # Create a mask containing only the largest component
    mask = np.uint8(labels == largest_component_index) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = np.vstack((contours))

    # Find the bounding box that contains all components#
    x, y, w, h = cv2.boundingRect(contours)
    rectangle_offset = 10
    x = x - rectangle_offset
    y = y - rectangle_offset
    w = w + rectangle_offset*2
    h = h + rectangle_offset*2

    # center, radius = cv2.minEnclosingCircle(contours)
    # radius = radius + 10
    # # Create an initial circular guess with 100 sample points
    # theta = np.linspace(0, 2 * np.pi, 50)
    # x = center[0] + radius * np.cos(theta)
    # y = center[1] + radius * np.sin(theta)
    # circle = np.column_stack((y, x))
    # Create an initial rectangular guess that contains all components
    rect = np.array([[y, x], [y + h, x], [y + h, x + w], [y, x + w], [y, x]])
    rect = rectangle(rect, num_points=50)
    #plt.plot(rect[:, 1], rect[:, 0], '.r')
    # Active contour parameters
    alpha = 0.15  # Weight of the contour energy term
    beta = 2.0  # Weight of the image energy term
    gamma = 0.001
    iterations = 1000  # Number of iterations

    # Perform active contour segmentation
    snake = active_contour(gaussian(mask), rect, alpha=alpha, beta=beta, gamma=gamma, max_num_iter=iterations)

    #plt.plot(snake[:, 1], snake[:, 0], '.g')
    #plt.show()

    rr, cc = polygon(snake[:, 1], snake[:, 0])
    mask[cc-1, rr-1] = 255
    mask = cv2.erode(mask, structuring_element)
    mask = mask / 255
    outputs = torch.unsqueeze(torch.tensor(mask), dim=0).to(dtype=torch.long)
    outputs = outputs.to(device)
    return outputs
