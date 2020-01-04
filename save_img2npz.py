import cv2
import numpy as np
import pandas as pd
import os
import sys
import time
import matplotlib.image as mpimg
from glob import glob1

def findnamefromid(id_):
    """ Convert sequence id into a file name."""
    id_arr = id_.split('#')
    return (id_arr[0][-2:]+'_'+id_arr[1]+'_'+'R'+id_arr[2]+
            '_'+'PICT'+id_arr[3].zfill(4)+'.JPG')

def main():
    if len(sys.argv) != 3:
        print("This script is executed by save_img2npz.py [image dir] [label path]")
        sys.exit()
        
    imagedir = sys.argv[1]
    labelpath = sys.argv[2]
    img_labels = pd.read_csv(labelpath)
    label_names = np.array(list(img_labels))[1:] # category corresponding to each number
    label_arr = img_labels.to_numpy()

    width = 320
    height = 240
    dim = (width, height)

    label_keys = dict()
    for label_list in label_arr: # key for each image file name
        label_keys[findnamefromid(label_list[0])] = np.argmax(label_list[1:])

    filelist = list() # find all image files under datadir
    for (dirpath, dirnames, filenames) in os.walk(imagedir):
        filelist += [os.path.join(dirpath, file) for file in filenames]
        
    cnt = 0
    index = 0
    len1GB = int(1e9/(1*height*width))
    num_block = 0

    for fname in filelist:
        if fname.split('/')[-1] in label_keys:
            cnt += 1
            if num_block*len1GB < cnt:
                if num_block == 0:
                    x = np.zeros([height, width, len1GB], dtype=np.uint8)
                else:
                    np.savez(imagedir.split('/')[-1]+'_block'+str(num_block), x=x, labels=labels)
                    x = np.zeros([height, width, len1GB], dtype=np.uint8)
                labels = list()
                num_block += 1
                index = 0 
                print('***created block '+str(num_block)+'***')
                sys.stdout.flush()

            img = mpimg.imread(fname)
            resized = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC).mean(axis=2).reshape(height, width)
            x[:, :, index] = resized
            index += 1
            labels += [label_keys[fname.split('/')[-1]]]
            
    if index != 0:
        x = x[:, :, :index]
        np.savez(imagedir.split('/')[-1]+'_block'+str(num_block), x=x, labels=labels)
        

if __name__ == "__main__":
    main()