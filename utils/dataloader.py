import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class HydrocarbonDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, augment=False, augmentation=None,  num_augmented_samples=5):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.augment = augment
        self.num_augmented_samples = num_augmented_samples
        # self.augmentation = transforms.Compose([
        #     transforms.RandomRotation(40, fill=(255,)),
        #     transforms.RandomResizedCrop(300, scale=(0.8, 1.0)),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor()
        # ])
        self.augmentation = augmentation

    def __len__(self):
        if self.augment:
            return len(self.data_frame) * (self.num_augmented_samples + 1)
        return len(self.data_frame)

    def __getitem__(self, idx):
        original_idx = idx % len(self.data_frame)
        img_name = os.path.join(self.root_dir, f"{self.data_frame.iloc[original_idx, 0]}.png")
        image = Image.open(img_name).convert('L')  # Convert to grayscale
        labels = self.data_frame.iloc[original_idx, 1:].values.astype('float')
        labels = torch.tensor(labels, dtype=torch.float32)

        if self.augment and idx >= len(self.data_frame):
            image = self.augmentation(image)
        elif self.transform:
            image = self.transform(image)

        return image, labels

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor()
])

csv_dir = 'data/hydrocarbon/CH_dict.csv'
image_dir = 'data/hydrocarbon/images'

# Create the dataset and dataloader
# dataset = HydrocarbonDataset(csv_file=csv_dir,
#                              root_dir=image_dir,
#                              transform=transform,
#                              augment=True,  # Enable augmentation
#                              num_augmented_samples=10)  # Number of augmented samples per original image

# _augmentation = transforms.Compose([
#     transforms.RandomRotation(40, fill=(255,)),
#     transforms.RandomResizedCrop(300, scale=(0.8, 1.0)),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor()
# ])

class ReplaceColorTransform:
    def __init__(self, target_color, replacement_color):
        self.target_color = target_color
        self.replacement_color = replacement_color

    def __call__(self, img):
        data = img.getdata()
        new_data = []
        for item in data:
            if item == self.target_color:
                new_data.append(self.replacement_color)
            else:
                new_data.append(item)
        img.putdata(new_data)
        return img

target_color = 245
replacement_color = 255

replace_color_transform = ReplaceColorTransform(target_color, replacement_color)

_augmentation = transforms.Compose([
    replace_color_transform,
    transforms.RandomRotation(40, fill=255),
    transforms.RandomResizedCrop(300, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

dataset = HydrocarbonDataset(csv_file=csv_dir,
                             root_dir=image_dir,
                             transform=transform,
                             augment=True,  # Enable augmentation
                             augmentation=_augmentation,
                             num_augmented_samples=10)  # Number of augmented samples per original image

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)