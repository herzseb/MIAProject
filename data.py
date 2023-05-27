import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import re

class SegmentationDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.labels = []
        self.file_list = self.get_file_list()
        self.transform = T.Compose([
        #T.RandomResizedCrop(image_size),
        #T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Grayscale(),
        ])
        

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        image_path = self.file_list[idx][0]
        mask_paths = self.file_list[idx][1]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        mean = torch.mean(image)
        std = torch.std(image)
        norm = T.Normalize(mean, std)
        image = norm(image)

        mask = torch.zeros_like(image)
        for mask_path in mask_paths:
            masklet = Image.open(mask_path)
            mask += self.transform(masklet)

        # Assign labels based on class names
        class_name = os.path.basename(os.path.dirname(image_path))
        if class_name == 'benign':
            label = 1
        elif class_name == 'malignant':
            label = 2
        elif class_name == 'normal':
            label = 0

        # Convert pixel values in the mask to class labels
        mask[mask > 0] = label

        return image, mask

    def get_file_list(self):
        pattern = r".*?mask_\d\.png"
        file_list = []
        for class_name in ['normal', 'benign', 'malignant']:
            class_dir = os.path.join(self.data_dir, class_name)
            names = os.listdir(class_dir)
            names.sort()
            for file_name in names:
                if file_name.endswith('.png') and not file_name.endswith('_mask.png') and not re.match(pattern, file_name):
                    image_path = os.path.join(class_dir, file_name)
                    mask_base = os.path.splitext(file_name)[0]
                    mask_paths = [os.path.join(class_dir, f'{mask_base}_mask.png')]
                    if os.path.exists(mask_paths[0]):
                        i = 1
                        while True:
                            additional_mask_path = os.path.join(class_dir, f'{mask_base}_mask{i}.png')
                            if not os.path.exists(additional_mask_path):
                                break
                            mask_paths.append(additional_mask_path)
                            i += 1
                    file_list.append((image_path, mask_paths))
                    self.labels.append(class_name)
        return file_list


def collate_fn(batch):
    images, masks = zip(*batch)

    # Get the maximum height and width among all images in the batch
    max_height = max(image.shape[1] for image in images)
    max_width = max(image.shape[2] for image in images)

    # Pad images and masks to match the maximum dimensions
    padded_images = []
    padded_masks = []
    for image, mask in zip(images, masks):
        height_diff = max_height - image.shape[1]
        width_diff = max_width - image.shape[2]
        padded_image = F.pad(image, (0, height_diff, width_diff, 0))
        padded_mask = F.pad(mask, (0, height_diff, width_diff, 0))
        padded_images.append(padded_image)
        padded_masks.append(padded_mask)

    # Convert the list of padded images and masks to tensors
    padded_images = torch.stack(padded_images)
    padded_masks = torch.stack(padded_masks)

    return padded_images, padded_masks
