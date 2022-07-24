import numpy as np
from math import log10


def psnr(img1, img2):
    """
    test PSNR value from 2 tensor
    """
    mse = np.mean((img1/1.0 - img2/1.0) ** 2)
    if mse < 1.0e-10:
        return 100, 0
    return 10 * log10(255.0**2/mse), mse