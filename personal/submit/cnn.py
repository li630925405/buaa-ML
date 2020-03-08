from torch import nn
import os

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),  # 25
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),  # 25 50
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer5 = nn.Conv2d(64, 120, kernel_size=3, padding=1)

        self.fc = nn.Sequential(
            # 28x28 -> 26x26 -> 13x13 -> 11x11 -> 5x5
            nn.Linear(120 * 5 * 5, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        #print("--------------")
        #print(x.shape)
        x = self.layer1(x)
        #print(x.shape)
        x = self.layer2(x)
        #print(x.shape)
        x = self.layer3(x)
        #print(x.shape)
        x = self.layer4(x)
        #print(x.shape)
        # x = x.view(n, -1)    n: 转换后有几行
        # x:  (batchsize, channel, height, width)
        # x除batchsize外其余变成一维
        #print(x.shape)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.fc(x)
        return x
