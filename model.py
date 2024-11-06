import torch
from torch import nn
from torch.nn import Conv2d, ReLU, MaxPool2d, Flatten, Linear


class Identify(nn.Module):
    def __init__(self):
        super(Identify, self).__init__()
        self.model=nn.Sequential(
            Conv2d(in_channels=1,out_channels=32,kernel_size=3,stride=1,padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2,stride=2),
            Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2,stride=2),

            Flatten(),
            Linear(64*7*7,128),
            ReLU(),
            Linear(128,10)
        )
    def forward(self,input):
        input=self.model(input)
        return input

identify = Identify()
if __name__ == '__main__':
    #测试一下是否模型输入输出是否正确
    input = torch.ones((64, 1, 28, 28))
    print(input.shape)
    output = identify(input)
    print(output.shape)

