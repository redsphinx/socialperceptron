import numpy as np
import os


def safe_mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


