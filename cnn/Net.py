import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input_dim:tuple[int,int], num_classes:int):
        super().__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bnn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bnn2 = nn.BatchNorm2d(64)

        self.dropout = nn.Dropout(p=0.5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        h,w = input_dim
        self.fc1 = nn.Linear(in_features=64 * int(h/4) * int(w/4), out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x: torch.Tensor):
        out = self.conv1(x) # (batch_size, 32, 28, 28)
        out = self.bnn1(out)
        out = F.relu(out)
        out = self.pool(out) #(batch_size, 32, 14, 14)

        out = self.conv2(out) #(batch_size, 64, 14, 14)
        out = self.bnn2(out)
        out = F.relu(out)
        out = self.pool(out) #(batch_size, 64, 7, 7)

        h,w = self.input_dim
        out = out.reshape(-1, 64 * int(h/4) * int(w/4))

        out = self.fc1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out

