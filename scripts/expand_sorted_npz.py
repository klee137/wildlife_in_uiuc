#!/usr/bin/python3.6

import numpy as np
import argparse
import sys
from glob import glob1

def getOptions(args=sys.argv[1:]):
    """Parse the command arguments into appropriate variables.
    
    args:
        args: input arguments provided after the script file name
    returns:
        options: class containing various input variables
    """
    parser = argparse.ArgumentParser(description="Parses command.")
    parser.add_argument("--inputDir", required=True, help="data directory of npz files to be expanded.")
    parser.add_argument("--saveDir", required=True, help="directory where expanded files will be saved.")
    parser.add_argument("-v", "--verbose", dest='verbose',action='store_true', help="Verbose mode.")
    options = parser.parse_args(args)
    return options

def main():
    options = getOptions(sys.argv[1:])
    flist = glob1(options.inputDir, '*.npz')
    for fname in flist:
        x = np.load(options.inputDir+'/'+fname)['x']
        height, width, num_samp = x.shape
        for n in range(num_samp):
            np.savez(options.saveDir+'/'+fname[:-4]+'_'+str(n), x=x[:, :, n])

    # x = f['x']
    # height, width, total_samp = x.shape
    # labels = f['labels']
    # FILE_SIZE = 1e9 # keeping each file at 1 GB
    # SAMP_SIZE = int(FILE_SIZE // (height*width)) 


    # for label in np.unique(labels):
    #     num_samp = np.sum(labels==label)
    #     curr_x = x[:, :, label==labels] 

    #     file_index = 0 
    #     flist = numerical_sort(glob1(options.saveDir, 'label'+str(label)+'_*.npz'))

    #     if len(flist) > 0:
    #         file_index = int(flist[-1].split('_')[-1][:-4])
    #         prev_f = np.load(options.saveDir+'/'+flist[-1])
    #         prev_x = prev_f['x']
    #         curr_x = np.concatenate((curr_x, prev_x), axis=2)
        
    #     # need additional file to store data
    #     if curr_x.shape[2] > SAMP_SIZE:
    #         np.savez(options.saveDir+'/'+'label'+str(label)+'_'+str(file_index),
    #             x=curr_x[:, :, :SAMP_SIZE])
    #         np.savez(options.saveDir+'/'+'label'+str(label)+'_'+str(file_index+1),
    #             x=curr_x[:, :, SAMP_SIZE:])
    #     else:
    #         np.savez(options.saveDir+'/'+'label'+str(label)+'_'+str(file_index),
    #             x=curr_x)




if __name__ == "__main__":
    main()