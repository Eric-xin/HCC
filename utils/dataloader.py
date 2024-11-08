import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class HydrocarbonDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, f"{self.data_frame.iloc[idx, 0]}.png")
        image = Image.open(img_name).convert('L')  # Convert to grayscale
        labels = self.data_frame.iloc[idx, 1:].values.astype('float')
        # debug the model
        # print(f"image: {image.size}, labels: {labels}")
        labels = torch.tensor(labels, dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
        
        return image, labels

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor()
])

dataset = HydrocarbonDataset(csv_file='data/hydrocarbon/CH_dict.csv', 
                             root_dir='data/hydrocarbon/images', 
                             transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)