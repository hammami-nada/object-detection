import torch.nn as nn #contains classes and functions for building neural networks

# CNN block definition
# this defines a custom CNN block as a subclass of (nn.Module)
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):  #defines tge contructor method
        # self is a reference to the current instance of the class
        # in_channels is the number of input channels to the convolutionam layer
        #**kwargs additionla keyword arguments which will be passsed to the nn.Conv2d constructor
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)  #,,.Conv2d is a convolutional layer
        self.batchnorm = nn.BatchNorm2d(out_channels) # nn.BatchNorm2d is a batch normalization
        self.leakyrelu = nn.LeakyReLU(0.1)  # LeakyReLU activation function

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))