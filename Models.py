import torch.nn as nn
from torch import flatten
###
#LeNet
###

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1756920,84),
            nn.ReLU(),
            #nn.Linear(84,1)
            nn.Linear(84,2)
        )

    def forward(self, x):
        out = self.main(x)
        return out

###
#AlexNet
###
class AlexNet(nn.Module):
    def __init__(self,dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,64,kernel_size=11,stride=4,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(64,192,kernel_size=5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(192,384,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6,6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,1)
        )

    def forward(self,x):
        x = self.features(x)
        x = self.avgpool(x)
        x = flatten(x,1)
        x = self.classifier(x)
        return x

###
#FNO
###
#Fourier layer

#Full FNO
