import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class EffNetB0(nn.Module):
    def __init__(self):
        super(EffNetB0, self).__init__()
        self.effnet_stage1 = nn.Conv2d(1, 32, 3, padding=1, stride=2) # in channel x out channel x kernel size 
        
        self.effnet_stage2 = nn.Sequential(nn.Conv2d(32, 32*1, 1),
                                           nn.ReLU6(),
                                           nn.Conv2d(32*1, 32*1, 3, padding=1, groups=32*1),
                                           nn.ReLU6(),
                                           nn.Conv2d(32*1, 16, 1))
        
        self.effnet_stage3 = nn.Sequential(nn.Conv2d(16, 16*6, 1),
                                           nn.ReLU6(),
                                           nn.Conv2d(16*6, 16*6, 3, padding=1, stride=2, groups=16*6),
                                           nn.ReLU6(),
                                           nn.Conv2d(16*6, 24, 1),
                                           nn.Conv2d(24, 24*6, 1),
                                           nn.ReLU6(),
                                           nn.Conv2d(24*6, 24*6, 3, padding=1, groups=24*6),
                                           nn.ReLU6(),
                                           nn.Conv2d(24*6, 24, 1))
        
        self.effnet_stage4 = nn.Sequential(nn.Conv2d(24, 24*6, 1),
                                           nn.ReLU6(),
                                           nn.Conv2d(24*6, 24*6, 5, padding=2, stride=2, groups=24*6),
                                           nn.ReLU6(),
                                           nn.Conv2d(24*6, 40, 1),
                                           nn.Conv2d(40, 40*6, 1),
                                           nn.ReLU6(),
                                           nn.Conv2d(40*6, 40*6, 5, padding=2, groups=40*6),
                                           nn.ReLU6(),
                                           nn.Conv2d(40*6, 40, 1))
        
        self.effnet_stage5 = nn.Sequential(nn.Conv2d(40, 40*6, 1),
                                           nn.ReLU6(),
                                           nn.Conv2d(40*6, 40*6, 3, padding=1, groups=40*6),
                                           nn.ReLU6(),
                                           nn.Conv2d(40*6, 80, 1),
                                           nn.Conv2d(80, 80*6, 1),
                                           nn.ReLU6(),
                                           nn.Conv2d(80*6, 80*6, 3, padding=1, groups=80*6),
                                           nn.ReLU6(),
                                           nn.Conv2d(80*6, 80, 1),
                                           nn.Conv2d(80, 80*6, 1),
                                           nn.ReLU6(),
                                           nn.Conv2d(80*6, 80*6, 3, padding=1, groups=80*6),
                                           nn.ReLU6(),
                                           nn.Conv2d(80*6, 80, 1))

        self.effnet_stage6 = nn.Sequential(nn.Conv2d(80, 80*6, 1),
                                           nn.ReLU6(),
                                           nn.Conv2d(80*6, 80*6, 5, padding=2, stride=2, groups=80*6),
                                           nn.ReLU6(),
                                           nn.Conv2d(80*6, 112, 1),
                                           nn.Conv2d(112, 112*6, 1),
                                           nn.ReLU6(),
                                           nn.Conv2d(112*6, 112*6, 5, padding=2, groups=112*6),
                                           nn.ReLU6(),
                                           nn.Conv2d(112*6, 112, 1),
                                           nn.Conv2d(112, 112*6, 1),
                                           nn.ReLU6(),
                                           nn.Conv2d(112*6, 112*6, 5, padding=2, groups=112*6),
                                           nn.ReLU6(),
                                           nn.Conv2d(112*6, 112, 1))
        
        self.effnet_stage7 = nn.Sequential(nn.Conv2d(112, 112*6, 1),
                                           nn.ReLU6(),
                                           nn.Conv2d(112*6, 112*6, 5, padding=2, stride=2, groups=112*6),
                                           nn.ReLU6(),
                                           nn.Conv2d(112*6, 192, 1),
                                           nn.Conv2d(192, 192*6, 1),
                                           nn.ReLU6(),
                                           nn.Conv2d(192*6, 192*6, 5, padding=2, groups=192*6),
                                           nn.ReLU6(),
                                           nn.Conv2d(192*6, 192, 1),
                                           nn.Conv2d(192, 192*6, 1),
                                           nn.ReLU6(),
                                           nn.Conv2d(192*6, 192*6, 5, padding=2, groups=192*6),
                                           nn.ReLU6(),
                                           nn.Conv2d(192*6, 192, 1),
                                           nn.Conv2d(192, 192*6, 1),
                                           nn.ReLU6(),
                                           nn.Conv2d(192*6, 192*6, 5, padding=2, groups=192*6),
                                           nn.ReLU6(),
                                           nn.Conv2d(192*6, 192, 1))
        
        self.effnet_stage8 = nn.Sequential(nn.Conv2d(192, 192*6, 1),
                                           nn.ReLU6(),
                                           nn.Conv2d(192*6, 192*6, 3, padding=1, groups=192*6),
                                           nn.ReLU6(),
                                           nn.Conv2d(192*6, 320, 1))
        
        self.effnet_stage9 = nn.Sequential(nn.Conv2d(320, 160, 1),
                                           nn.MaxPool2d(3, stride=2),
                                           Flatten(),
                                           nn.Linear(1920, 1000),
                                           nn.Dropout(p=0.2),
                                           nn.Linear(1000, 200),
                                           nn.Dropout(p=0.2),
                                           nn.Linear(200, 54),
                                           nn.Dropout(p=0.2),
                                           nn.Softmax(dim=1))
    def forward(self, x):
        x = self.effnet_stage1(x)
        x = self.effnet_stage2(x)
        x = self.effnet_stage3(x)
        x = self.effnet_stage4(x)
        x = self.effnet_stage5(x)
        x = self.effnet_stage6(x)
        x = self.effnet_stage7(x)
        x = self.effnet_stage8(x)
        x = self.effnet_stage9(x)        
        
        return x

class Print(nn.Module):
    def forward(self, x):
        print(x.size())
        
        return x
    
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(-1, self.num_flat_features(x))
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
            
        return num_features   
