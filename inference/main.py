from datetime import datetime
import logging
import multiprocessing
from pathlib import Path
import cv2
import numpy as np
from PIL import ImageFile
import pandas as pd
import torch
from assets.efficient_net_b0 import EffNetB0
import sys


# We get to see the log output for our execution, so log away!
logging.basicConfig(level=logging.INFO)

ASSET_PATH = Path(__file__).parents[0] / "assets"
MODEL_PATH = ASSET_PATH / "effnet0-v1.pth"

# The images will live in a folder called 'data' in the container
DATA_PATH = Path(__file__).parents[0] / "data"


def perform_inference():
    """This is the main function executed at runtime in the cloud environment. """
    logging.info("Loading model.")
    model = EffNetB0()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    logging.info("Loading and processing metadata.")

    # Our preprocessing selects the first image for each sequence
    test_metadata = pd.read_csv(DATA_PATH / "test_metadata.csv", index_col="seq_id")
    test_metadata = (
        test_metadata.sort_values("file_name").groupby("seq_id").first().reset_index()
    )
    

    # Prepend the path to our filename since our data lives in a separate folder
    test_metadata["full_path"] = test_metadata.file_name.map(
        lambda x: str(DATA_PATH / x)
    )
    img_paths = list(test_metadata["full_path"].values.flatten())
    num_img = len(img_paths)
    # print(test_metadata["full_path"])

    logging.info("Starting inference.")

    # Preallocate prediction output
    submission_format = pd.read_csv(DATA_PATH / "submission_format.csv", index_col=0)
    num_labels = submission_format.shape[1]
    output = np.zeros((test_metadata.shape[0], num_labels))

    height, width = 240, 320
    dim = (width, height)

    batch_size = 3 # need to be changed 1/8/2020
    steps = num_img // batch_size
    step = 0
    

    while True:
        # process data into tensors
        img_batch = img_paths[step*batch_size:(step+1)*batch_size]
        print(img_batch)
        x = np.empty((len(img_batch), 1, height, width), dtype=np.float32)
        for i, fname in enumerate(img_batch):
            img = cv2.imread(fname)
            resized = cv2.resize(img, dim,
                interpolation=cv2.INTER_CUBIC).mean(axis=2).reshape(height, width)
            x[i, 0, :, :] = resized
        # normalize x
        x = ((x-x.mean(axis=(2, 3)).reshape(len(img_batch), 1, 1, 1))/
            x.std(axis=(2, 3)).reshape(len(img_batch), 1, 1, 1))

        y = model(torch.from_numpy(x)).data.numpy()
        print(y.shape)
        output[step*batch_size:(step+1)*batch_size, :] = y


        step += 1
        if step*batch_size >= num_img:
            break
    
    my_submission = pd.DataFrame(
        np.stack(output),
        # Remember that we are predicting at the sequence, not image level
        index=test_metadata.seq_id,
        columns=submission_format.columns,
    )

    # We want to ensure all of our data are floats, not integers
    my_submission = my_submission.astype(np.float)

    # Save out submission to root of directory
    my_submission.to_csv("submission.csv", index=True)
    logging.info(f"Submission saved.")


if __name__ == "__main__":
    perform_inference()
