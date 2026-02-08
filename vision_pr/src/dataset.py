import torch 
import pandas as pd
import numpy as np 
from torch.utils.data import Dataset 
from PIL import Image

class FER2013Dataset(Dataset):
    def __init__(self, csv_file, split='Training', transform=None):
        """
        Args:
            csv_file (string): Path to csv file
            split (string): 'Training', 'PublicTest', 'PrivateTest'
            transform (callable, optional): PyTorch transforms for augmentation
        """

        self.data = pd.read_csv(csv_file)

        if 'Usage' in self.data.columns:
            self.data = self.data[self.data['Usage'] == split]

        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]

        emotion = row['emotion']
        pixels_string = row['pixels']
        
        image_array = np.array(pixels_string.split(), dtype='uint8')
        image_array = image_array.reshape(48, 48)
        image = Image.fromarray(image_array)

        if self.transform:
            # Apply Transforms
            image = self.transform(image)

        return image, torch.tensor(emotion, dtype=torch.long)