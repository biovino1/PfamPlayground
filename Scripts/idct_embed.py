"""================================================================================================
This script accepts a directory of embeddings and returns their inverse discrete cosine transforms.

Ben Iovino  07/04/23   SearchEmb
================================================================================================"""

import argparse
import numpy as np
import os
from scipy.fft import dct, idct
import logging

logging.basicConfig(filename='Data/idct_embed.log',
                     level=logging.INFO, format='%(message)s')


def scale(v: np.ndarray) -> np.ndarray:
    """=============================================================================================
    Scale from protsttools. Takes a vector and returns it scaled between 0 and 1.

    :param v: vector to be scaled
    :return: scaled vector
    ============================================================================================="""

    M = np.max(v)
    m = np.min(v)
    return (v - m) / float(M - m)


def iDCTquant(v: np.ndarray, n: int) -> np.ndarray:
    """=============================================================================================
    iDCTquant from protsttools. Takes a vector and returns its inverse discrete cosine transform.

    :param v: vector to be transformed
    :param n: number of coefficients to keep
    :return: transformed vector
    ============================================================================================="""

    f = dct(v.T, type=2, norm='ortho')
    trans = idct(f[:,:n], type=2, norm='ortho')
    for i in range(len(trans)):  #pylint: disable=C0200
        trans[i] = scale(trans[i])
    return trans.T

def main():
    """=============================================================================================
    Main takes a directory of embeddings
    ============================================================================================="""

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='Data/prott5_embed')
    args = parser.parse_args()

    # Go through embeddings and call idct on each one
    direc = args.d
    for fam in os.listdir(direc):
        for emb in os.listdir(direc + '/' + fam):
            emb_path = direc + '/' + fam + '/' + emb
            embed = np.load(emb_path)
            embed = idct(embed, norm='ortho')
            np.save(emb_path, embed)

            logging.info('iDCT performed on %s...', emb_path)


if __name__ == '__main__':
    main()
