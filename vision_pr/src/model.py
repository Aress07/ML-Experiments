import torch.nn as nn
import torch.nn.functional as F

class ClassicEmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            # 1, 48, 48
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            # 32, 24, 24 
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ELU(), 
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            # 64, 12, 12
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            # 128, 6, 6
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 6 * 6, 512),
            nn.ELU(),
            nn.Linear(512, 64),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x