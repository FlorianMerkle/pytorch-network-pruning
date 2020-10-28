import torch.nn as nn

from . custom_modules import MaskedLinearLayer, MaskedConvLayer, ResBlock

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.c1 = MaskedConvLayer((64, 3, 7, 7), padding=(3,3),bias=False, stride=2)
        self.p1 = nn.MaxPool2d((3,3), stride=(2,2), padding=(1))
        self.r1 = ResBlock(64,64,padding=1)
        self.r2 = ResBlock(64,64,padding=1)
        self.r3 = ResBlock(64,128,stride=2,padding=1)
        self.r4 = ResBlock(128,128,padding=1)
        self.r5 = ResBlock(128,256,stride=2,padding=1)
        self.r6 = ResBlock(256,256,padding=1)
        self.r7 = ResBlock(256,512,stride=2,padding=1)
        self.r8 = ResBlock(512,512,padding=1)
        self.p2 = nn.AvgPool2d(7)
        self.d1 = MaskedLinearLayer((512,10), activation=None)
    
    def forward(self, inputs):
        x = self.c1(inputs)
        x = self.p1(x)
        x = self.r1(x)
        x = self.r2(x)
        x = self.r3(x)
        x = self.r4(x)
        x = self.r5(x)
        x = self.r6(x)
        x = self.r7(x)
        x = self.r8(x)
        x = self.p2(x)
        x = x.view(x.shape[0], x.shape[1])
        x = self.d1(x)
        return (x)
        
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.c1 = MyConvLayer((6, 1, 5, 5), activation='relu')
        self.c2 = MyConvLayer((16, 6, 5, 5), activation='relu')
        self.fc1 = MyLinearLayer((256, 128))
        self.fc2 = MyLinearLayer((128, 84))
        self.fc3 = MyLinearLayer((84, 10), activation=None)
        
    def forward(self, inputs):
        x = self.c1(inputs)
        x = nn.AvgPool2d(2,2)(x)
        x = self.c2(x)
        x = nn.AvgPool2d(2,2)(x)
        x = x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x