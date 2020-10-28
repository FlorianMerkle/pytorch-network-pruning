import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedLinearLayer(nn.Module):
    """ Custom Linear layer with pruning mask"""
    def __init__(self, shape, bias=True, activation='relu'):
        super(MaskedLinearLayer, self).__init__()
        self.b, self.a = bias, activation
        weights = torch.empty(shape)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        mask = torch.ones(shape)
        self.mask = nn.Parameter(mask, requires_grad=False)
        if self.b == True:
            bias = torch.zeros(self.weights.shape[-1])
            self.bias = nn.Parameter(bias)

        # initialize weights and biases
        nn.init.xavier_uniform_(self.weights)

    def forward(self, inputs):
        x = torch.mm(inputs, self.weights*self.mask)
        if self.b == True:
            x = torch.add(x, self.bias)
        if self.a == 'relu':
            x = F.relu(x)
        return x

class MaskedConvLayer(nn.Module):
    """ Custom Conv layer with pruning mask"""
    def __init__(self, shape, bias=True, stride=1, padding=0, activation=None):
        super(MaskedConvLayer, self).__init__()
        self.b, self.s, self.p, self.a = bias, stride, padding, activation
        weights = torch.empty(shape)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        mask = torch.ones(shape)
        self.mask = nn.Parameter(mask, requires_grad=False)
        if self.b == True:
            bias = torch.zeros(self.weights.shape[0])
            self.bias = nn.Parameter(bias)

        # initialize weights and biases
        nn.init.xavier_uniform_(self.weights)

    def forward(self, inputs):
        x = F.conv2d(inputs, self.weights*self.mask, bias=None, stride=self.s, padding=self.p)
        if self.b == True:
            #reshape the bias
            b = self.bias.reshape((1, self.bias.shape[0], 1,1))
            x = torch.add(x, b)
        if self.a =='relu':
            x = F.relu(x)
        return x

class ResBlock(nn.Module):
    """ ResBlock made from masked Layers"""
    def __init__(self, input_channels, output_channels, padding=0 , stride=1, filter_size=3):
        super(ResBlock, self).__init__()
        self.s = stride
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.c1 = MaskedConvLayer((output_channels, input_channels, filter_size, filter_size), padding=padding, bias=False, stride=self.s)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.c2 = MaskedConvLayer((output_channels, output_channels, filter_size, filter_size), padding=padding, bias=False, stride=1)
        if self.s == 2:
            self.c3 = MaskedConvLayer((output_channels, input_channels, 1,1),padding=0, bias=False, stride=self.s)
        
    def forward(self, inputs):
        shortcut = inputs
        #print('inputs',shortcut.shape)
        x = self.c1(F.relu(self.bn1(inputs)))
        #print('after conv 1',x.shape)
        x = self.c2(F.relu(self.bn2(x)))
        #print('after conv 1',x.shape)
        if self.s == 2:
            shortcut = self.c3(shortcut)
        #print('shortcut',shortcut.shape)
        x = torch.add(x, shortcut)
        return x