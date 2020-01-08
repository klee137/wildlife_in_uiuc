#!/usr/bin/python3.6

import argparse
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from tqdm import tqdm

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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


def getOptions(args=sys.argv[1:]):
    """Parse the command arguments into appropriate variables.
    
    args:
        args: input arguments provided after the script file name
    returns:
        options: class containing various input variables
    """
    parser = argparse.ArgumentParser(description="Parses command.")
    parser.add_argument("--inputFile", help="data path (npz file) to be loaded.")
    parser.add_argument("--saveModel", help="path where model weights will be saved.")
    parser.add_argument("--loadModel", help="path where model weights are stored.")
    parser.add_argument("--numIter", type=int, help="Number of iteration a model will iterate through mini batch of size 64")
    parser.add_argument("-v", "--verbose",dest='verbose',action='store_true', help="Verbose mode.")
    options = parser.parse_args(args)
    return options

def main():
    options = getOptions(sys.argv[1:])
    if len(sys.argv) not in [2, 3, 4, 5, 6]:
        print("Wrong number of arguments. Use --help for more information. An example of execution would be\n ./train_effnet0.py --inputFile=SER_S1_block1.npz --saveModel=effnet0.pth --loadModel=effnet0.pth --numIter=1 -v")
        sys.exit()

    # trainnet --inputFile=SER_S1_block1.npz & trainnet --inputFile=SER_S1_block2.npz & trainnet --inputFile=SER_S1_block3.npz & trainnet --inputFile=SER_S1_block4.npz & trainnet
    #--inputFile=SER_S1_block5.npz & trainnet --inputFile=SER_S1_block6.npz
        
    # Load input data
    f = np.load(options.inputFile)
    x = f['x'].astype(np.float32)
    height, width, samp = x.shape
    # normalize x
    x = (x-x.mean(axis=(0, 1)).reshape(1, 1, samp))/x.std(axis=(0, 1)).reshape(1, 1, samp)

    labels = f['labels']
    # Define Efficient Net
    net = Net()
    if options.loadModel != None:
        net.load_state_dict(torch.load(options.loadModel))
        net.eval()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.004, momentum=0.9)
    
    # Train EfficientNet
    x = np.moveaxis(x, [2], [0]).reshape(x.shape[2], 1, height, width)
    
    batch_size = 64
    total_batch_size = x.shape[0] // batch_size
    index = 0 
    
    for i in tqdm(range(options.numIter)):
        prev_time = time.time()
        x_b = torch.from_numpy(x[batch_size*index:batch_size*(index+1)]).float()
        labels_b = torch.from_numpy(labels[batch_size*index:batch_size*(index+1)])

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(x_b)
        loss = criterion(outputs, labels_b)
        loss.backward()
        optimizer.step()
        # print statistics
        if options.verbose:
            print('[{0:03d}] loss: {1:6.8f}'.format(i+1, loss.item()/2000.))
            print('time passed for one batch: {}'.format(time.time()-prev_time))
        if index >= total_batch_size:
            index = 0
        else:
            index += 1
        
    torch.save(net.state_dict(), options.saveModel)
    print('Finished Training')







if __name__ == "__main__":
    main()
