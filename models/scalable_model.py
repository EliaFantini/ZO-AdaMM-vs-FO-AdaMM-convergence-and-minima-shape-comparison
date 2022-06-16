from torch import nn
import torch
import torch.nn.functional as F
from math import sqrt

# Example of a scale to get the parameters of networks between 1000 and 500000
# start = 1./9
# scale = torch.linspace(start,30,50)
    
class ModularModel(nn.Module):
    def __init__(self,scale, init=False):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 3, 3)
        self.fc1 = nn.Linear(3 * 5 * 5, max(10,int(scale*120)))
        self.fc2 = nn.Linear(max(10,int(scale*120)), max(10,int(sqrt(scale*120))))
        self.fc3 = nn.Linear(max(10,int(sqrt(scale*120))), 10)

        if init:
            # Initialize the weights and bias of the model
            nn.init.xavier_normal_(self.conv1.weight)
            nn.init.xavier_normal_(self.conv2.weight)
            nn.init.xavier_normal_(self.fc1.weight)
            nn.init.xavier_normal_(self.fc2.weight)
            nn.init.xavier_normal_(self.fc3.weight)

            self.conv1.bias.data.fill_(0.01)
            self.conv2.bias.data.fill_(0.01)
            self.fc1.bias.data.fill_(0.01)
            self.fc2.bias.data.fill_(0.01)
            self.fc3.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x