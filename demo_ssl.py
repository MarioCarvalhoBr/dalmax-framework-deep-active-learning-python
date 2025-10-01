import os
import time
from PIL import Image

import numpy as np
import torch

from demo_lda_vctex import extract_features_from_folder
from utils.LOGGER import get_logger, get_path_logger
import core.tools.SSL.code_kmh as code_kmh


# Global logger
logger = get_logger()
path_logger = get_path_logger()

def foo():
    print(f"\n\nHello from SSL demo_ssl.py\n")


def main():
    foo()
    
if __name__ == "__main__":
    main()