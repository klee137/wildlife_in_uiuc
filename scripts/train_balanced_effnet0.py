#!/usr/bin/python3.6

import argparse
import sys
sys.path.append('..')
import time
import os
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from model import EffNetB0
from glob import glob1

def getOptions(args=sys.argv[1:]):
    """Parse the command arguments into appropriate variables.
    
    args:
        args: input arguments provided after the script file name
    returns:
        options: class containing various input variables
    """
    parser = argparse.ArgumentParser(description="Parses command.")
    parser.add_argument("--inputDir", required=True, help="data directory.")
    parser.add_argument("--weightPath", required=True, help="path where model weights will be saved/loaded.")
    parser.add_argument("--numIter", type=int, required=True, help="Number of iteration a model will iterate \
                                                     through each labeled data (54 total).")
    parser.add_argument("-v", "--verbose",dest='verbose',action='store_true', help="Verbose mode.")
    options = parser.parse_args(args)
    return options

def main():
    options = getOptions(sys.argv[1:])

    # trainnet --inputFile=SER_S1_block1.npz & trainnet --inputFile=SER_S1_block2.npz & trainnet --inputFile=SER_S1_block3.npz & trainnet --inputFile=SER_S1_block4.npz & trainnet
    #--inputFile=SER_S1_block5.npz & trainnet --inputFile=SER_S1_block6.npz
    width = 320
    height = 240
    num_labels = 54 
    # Define Efficient Net
    net = EffNetB0()
    if os.path.exists(options.weightPath):
        net.load_state_dict(torch.load(options.weightPath))
        net.eval()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.004, momentum=0.9)

    data_labels = []
    # one time sort
    for label in range(num_labels):
        file_prefix = 'label' + str(label)
        flist = glob1(options.inputDir, file_prefix+'*')
        random.shuffle(flist)
        data_labels += [flist]

    for i in tqdm(range(options.numIter)):
        minibatch = np.empty((num_labels, 1, height, width), dtype=np.float32)
        for label in range(num_labels):
            fname = random.choice(data_labels[label])
            f = np.load(options.inputDir+'/'+fname)
            x = f['x']
            minibatch[label, 0, :, :] = x
        
        # normalize minibatch
        minibatch = ((minibatch-minibatch.mean(axis=(2, 3)).reshape(num_labels, 1, 1, 1))/
                      minibatch.std(axis=(2, 3)).reshape(num_labels, 1, 1, 1))
        prev_time = time.time()
        x_batch = torch.from_numpy(minibatch)
        labels_batch = torch.from_numpy(np.arange(num_labels))

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(x_batch)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()
        # print statistics
        if options.verbose:
            print('[{0:03d}] loss: {1:6.8f}'.format(i+1, loss.item()/2000.))
            print('time passed for one batch: {}'.format(time.time()-prev_time))
        
    torch.save(net.state_dict(), options.weightPath)
    print('Finished Training')


if __name__ == "__main__":
    main()
