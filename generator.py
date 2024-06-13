import torch
import torch.nn as nn
import numpy as np
import mlflow

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Conditioning(nn.Module):
    """
    When called, takes the context images to be given to the generator and downsamples them 
    Halves spatial dim every conv2d (800*800 -> 400*400 ...) and doubles depth (60 -> 120 ...)
    """

    def __init__(self):
        super().__init__()

        # LAYERS TODO maybe make this less hardcoded 
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=60,kernel_size= 3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels=60, out_channels=120,kernel_size= 3, padding = 1)
        self.conv3 = nn.Conv2d(in_channels=120, out_channels=240,kernel_size= 3, padding = 1)
        self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, context):
        # print(context.shape)
        x = self.conv1(context)
        x = self.relu(x)
        context3 = self.pooling(x)
        # print(x.shape)
        x = self.conv2(context3)
        x = self.relu(x)
        context2 = self.pooling(x)
        # print(x.shape)
        x = self.conv3(context2)
        x = self.relu(x)
        context1 = self.pooling(x)
        # print(x.shape)
        return context1, context2, context3 # 240, 120 and 60 dims respectively

class Generator(nn.Module):
    """
    When called, as Generator(input), the input will be a batch of satellite sequences. 
    The output will be the predicted images (which will be concatenated onto the context sequences)
    """

    def __init__(self):
        super().__init__()

        # PARAMS
        self.latent_channels = 100

        self.latent_length = 100 * 100 * self.latent_channels # 200 latent channels of same dims (100x100) as downsampled context
        self.context_downsampler = Conditioning()
        
        # DEFINE LAYERS
        self.convtransp1 = nn.Conv2d(in_channels=240+self.latent_channels, out_channels=220,kernel_size= 3, padding = 1)
        self.convtransp2 = nn.Conv2d(in_channels=220+120, out_channels=110,kernel_size= 3, padding = 1)
        self.convtransp3 = nn.Conv2d(in_channels=110+60, out_channels=55,kernel_size= 3, padding = 1)
        self.convtransp4 = nn.Conv2d(in_channels=55, out_channels=1,kernel_size= 3, padding = 1)
        self.norm1 = nn.BatchNorm2d(220)
        self.norm2 = nn.BatchNorm2d(110)
        self.norm3 = nn.BatchNorm2d(55)
        self.norm4 = nn.BatchNorm2d(1)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.leaky_relu = nn.LeakyReLU()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, context):
        batch_size = context.shape[0]

        # create latent vector
        latent = np.random.normal(0,1,self.latent_length * batch_size) # generate random (1D) vector of length compatible with context dims
        latent = torch.from_numpy(latent).type(torch.FloatTensor).reshape((batch_size, self.latent_channels,100,100)).to(device)

        # create context stack
        context1, context2, context3 = self.context_downsampler(context) # 240, 120 and 60 dims respectively

        # upsampling block 1
        latent_context1 = torch.cat((latent, context1), dim=1) # merge latent and context -> tensor to be upsampled

        x = self.convtransp1(latent_context1)
        x = self.leaky_relu(x)
        # x = self.norm1(x)
        x = self.upsample(x)

        # upsampling block 2
        upsampled1_context2 = torch.cat((x, context2), dim=1)
        x = self.convtransp2(upsampled1_context2)
        x = self.leaky_relu(x)
        # x = self.norm2(x)
        x = self.upsample(x)

        # upsampling block 3
        upsampled2_context3 = torch.cat((x, context3), dim=1)
        x = self.convtransp3(upsampled2_context3)
        x = self.leaky_relu(x)
        # x = self.norm3(x)
        x = self.upsample(x)

        # upsampling block 4
        x = self.convtransp4(x)
        x = self.leaky_relu(x)
        # x = self.norm4(x)

        # CONCATENATE CONTEXT AND GENERATED IMAGE -> 'FAKE' SEQUENCE
        pred_sequence = torch.cat((context, x), dim=1)

        return pred_sequence
