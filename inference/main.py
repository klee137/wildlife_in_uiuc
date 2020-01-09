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




if __name__ == "__main__":
    perform_inference()
