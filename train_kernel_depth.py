import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
import os
from torch.utils.data import Dataset, DataLoader

class FullHologramCNNWithActivationPooling(nn.Module):
    def __init__(self, hologram_size):
        super(FullHologramCNNWithActivationPooling, self).__init__()
        self.hologram_size = hologram_size
        # Convolution with a kernel the size of the hologram
        self.conv1 = nn.Conv2d(1, 1, kernel_size=hologram_size, stride=hologram_size, bias=None)
        # Activation function
        self.activation = nn.ReLU()
        
    def forward(self, x):
        # Ensure input is the correct size
        x = F.pad(x, (0, self.hologram_size[1] - x.size(3), 0, self.hologram_size[0] - x.size(2)))
        x = self.conv1(x)
        x = self.activation(x)
        # "Pooling" through interpolation to adjust the output size back to the original dimensions
        x = F.interpolate(x, size=self.hologram_size, mode='bilinear', align_corners=False)
        return x


class ComplexHolographyCNN(nn.Module):
    def __init__(self):
        super(ComplexHolographyCNN, self).__init__()
        # First convolutional layer with same padding to maintain output size
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        # Pooling layer
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        # Another pooling layer
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        # Upsampling layers
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv1 = nn.ConvTranspose2d(32, 16, kernel_size=5, padding=2)
        
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(16, 1, kernel_size=5, padding=2)

    def forward(self, x):
        # Downsample
        x = self.relu1(self.conv1(x))
        x, indices1 = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x, indices2 = self.pool2(x)
        
        # Upsample
        x = self.unpool1(x, indices2)
        x = self.deconv1(x)
        x = self.unpool2(x, indices1)
        x = self.deconv2(x)
        
        return x


class HologramPairDataset(Dataset):
    def __init__(self, folder_path_hologram1, folder_path_hologram2, transform=None):
        """
        Initialize the dataset with paths to the folders containing the hologram images.
        Assumes that the files in both folders match and are sorted in order.
        """
        self.files_hologram1 = sorted([os.path.join(folder_path_hologram1, f) 
                                       for f in os.listdir(folder_path_hologram1) if f.endswith('.tif')])
        self.files_hologram2 = sorted([os.path.join(folder_path_hologram2, f) 
                                       for f in os.listdir(folder_path_hologram2) if f.endswith('.tif')])
        self.transform = transform

    def __len__(self):
        """Return the number of pairs in the dataset."""
        return min(len(self.files_hologram1), len(self.files_hologram2))

    def __getitem__(self, idx):
        """Load and return a pair of holograms at the specified index."""
        hologram1_path = self.files_hologram1[idx]
        hologram2_path = self.files_hologram2[idx]

        hologram1 = Image.open(hologram1_path).convert('L')
        hologram2 = Image.open(hologram2_path).convert('L')

        if self.transform:
            hologram1 = self.transform(hologram1)
            hologram2 = self.transform(hologram2)

        return hologram1, hologram2


transform = transforms.Compose([
    transforms.ToTensor(),
    # Add any other transformations here
    # Example: transforms.Normalize(mean=[0.5], std=[0.5])
])

hologram_dataset = HologramPairDataset(
    folder_path_hologram1='C:/Users/jiaqi/Documents/GitHub/Imcov_DIH/train/holo1',
    folder_path_hologram2='C:/Users/jiaqi/Documents/GitHub/Imcov_DIH/train/holo2',
    transform=transform
)

data_loader = DataLoader(hologram_dataset, batch_size=4, shuffle=True, num_workers=0)

# Assuming the model, loss function, and optimizer are defined
model = ComplexHolographyCNN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10
for epoch in range(num_epochs):
    for hologram1, hologram2 in data_loader:
        # Process the batch
        optimizer.zero_grad()
        output = model(hologram1)
        loss = criterion(output, hologram2)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')


trained_kernel = model.conv1.weight.data.numpy()
print(trained_kernel)
