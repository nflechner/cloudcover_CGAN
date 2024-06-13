import torch.nn as nn
import torch
from math import prod
from torch.nn.utils.parametrizations import spectral_norm

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.temporal_discriminator = Temporal_Discriminator()
        self.spatial_discriminator = Spatial_Discriminator()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        temporal_predictions = self.temporal_discriminator(x)
        spatial_predictions = self.spatial_discriminator(x[:,4,:,:].unsqueeze(dim=1))

        predictions = torch.cat((temporal_predictions, spatial_predictions))

        return predictions

class Temporal_Discriminator(nn.Module):
    "Classifies whether an sequence is real/fake"

    def __init__(self): # a list of hidden layer sizes for n hidden layers
        super().__init__()

        self.input_size = 5 * 800 * 800

        # define nn layers (TODO this is still way too simple, but to check no errors and increase speed)
        self.conv1 = nn.Conv2d(in_channels = 5, out_channels = 20, kernel_size= 3, stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 20, out_channels = 40, kernel_size= 3, stride = 2)
        self.conv3 = nn.Conv2d(in_channels = 40, out_channels = 80, kernel_size= 3, stride = 2)
        self.norm1 = nn.BatchNorm2d(20)
        self.norm2 = nn.BatchNorm2d(40)
        self.norm3 = nn.BatchNorm2d(80)
        self.pool = nn.MaxPool2d(3, stride = 3)
        self.linear1 = nn.Linear(720,360)
        self.linear2 = nn.Linear(360,1)

        # define nn functions 
        self.flatten = nn.Flatten(1,3) 
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()  # For binary classification
        self.tanh = nn.Tanh()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.norm1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.norm2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.norm3(x)

        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)

        x = self.linear2(x)
        x = self.sigmoid(x)
        # x = self.tanh(x)
        return x

class Spatial_Discriminator(nn.Module):
    "Classifies whether a satellite image is real/fake (versus whether the sequence is real/fake)"

    def __init__(self): # a list of hidden layer sizes for n hidden layers
        super().__init__()

        self.input_size = 800 * 800

        # define nn layers (TODO this is still way too simple, but to check no errors and increase speed)
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 10, kernel_size= 3, stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 10, out_channels = 40, kernel_size= 3, stride = 2)
        self.conv3 = nn.Conv2d(in_channels = 40, out_channels = 80, kernel_size= 3, stride = 2)
        self.norm1 = nn.BatchNorm2d(10)
        self.norm2 = nn.BatchNorm2d(40)
        self.norm3 = nn.BatchNorm2d(80)
        self.pool = nn.MaxPool2d(3, stride = 3)
        self.linear1 = nn.Linear(720,360)
        self.linear2 = nn.Linear(360,1)

        # define nn functions 
        self.flatten = nn.Flatten(1,3) 
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()  # For binary classification
        self.tanh = nn.Tanh()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.norm1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.norm2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.norm3(x)

        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)

        x = self.linear2(x)
        x = self.sigmoid(x)
        # x = self.tanh(x)
        return x
    
class Temp_spectral_norm(nn.Module):
    def __init__(self, channels_in):
        super().__init__()
        self.channels_in = channels_in
        
    def forward(self, x):
        return spectral_norm(nn.Linear(self.channels_in, 1))


class Spatial_spectral_norm(nn.Module):
    def __init__(self,channels_in):
        super().__init__()
        self.channels_in = channels_in
        
    def forward(self, x):
        return spectral_norm(nn.Linear(self.channels_in, 1))