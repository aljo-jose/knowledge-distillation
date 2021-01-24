import torch
import torch.nn as nn
import torch.nn.functional as F


class FCModel(nn.Module):
    def __init__(self):
        super(FCModel,self).__init__()
        self.lin1 = nn.Linear(28*28, 1200)
        self.lin2 = nn.Linear(1200,1200)
        self.lin3 = nn.Linear(1200,10)
        
    def forward(self, inputs):
        x = F.relu(self.lin1(inputs.view(-1,28*28)))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        out = F.log_softmax(x, dim=1)
        return out

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel,self).__init__()
        self.lin1 = nn.Linear(28*28, 400)
        self.lin2 = nn.Linear(400,10)
        
    def forward(self, inputs):
        x = F.relu(self.lin1(inputs.view(-1,28*28)))
        x = self.lin2(x)
        output = F.log_softmax(x, dim=1)
        return output