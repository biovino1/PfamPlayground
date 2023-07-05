"""================================================================================================
This script accepts a directory of embeddings and returns their inverse discrete cosine transforms.

Ben Iovino  07/04/23   SearchEmb
================================================================================================"""

import argparse
import os
import logging
import numpy as np
from scipy.fft import dct, idct

logging.basicConfig(filename='Data/idct_embed.log',
                     level=logging.INFO, format='%(message)s')


def scale(vec: np.ndarray) -> np.ndarray:
    """=============================================================================================
    Scale from protsttools. Takes a vector and returns it scaled between 0 and 1.

    :param v: vector to be scaled
    :return: scaled vector
    ============================================================================================="""

    maxi = np.max(vec)
    mini = np.min(vec)
    return (vec - mini) / float(maxi - mini)


def iDCTquant(vec: np.ndarray, num: int) -> np.ndarray:
    """=============================================================================================
    iDCTquant from protsttools. Takes a vector and returns its inverse discrete cosine transform.

    :param v: vector to be transformed
    :param n: number of coefficients to keep
    :return: transformed vector
    ============================================================================================="""

    f = dct(vec.T, type=2, norm='ortho')
    trans = idct(f[:,:num], type=2, norm='ortho')  #pylint: disable=E1126
    for i in range(len(trans)):  #pylint: disable=C0200
        trans[i] = scale(trans[i])  #pylint: disable=E1137
    return trans.T  #pylint: disable=E1101


def quant2D(emb: np.ndarray, n: int, m: int) -> np.ndarray:
    """=============================================================================================
    quant2D from protsttools. Takes an embedding and returns its inverse discrete cosine transform
    on both axes.

    :param emb: embedding to be transformed (n x m array)
    :param n: number of coefficients to keep on first axis
    :param m: number of coefficients to keep on second axis
    :return: transformed embedding (n*m 1D array)
    ============================================================================================="""

    dct = iDCTquant(emb[1:len(emb)-1],n)  #pylint: disable=W0621
    ddct = iDCTquant(dct.T,m).T
    ddct = ddct.reshape(n*m)
    return (ddct*127).astype('int8')


def main():
    """=============================================================================================
    Main takes a directory of embeddings and calls quant2D on each one to return its inverse
    discrete cosine transform.
    ============================================================================================="""

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='Data/prott5_embed')
    args = parser.parse_args()

    # Go through embeddings and call idct on each one
    direc = args.d
    for fam in os.listdir(direc):  # Each family in the directory
        for emb in os.listdir(direc + '/' + fam):  # Each emb in the family
            emb_path = direc + '/' + fam + '/' + emb

            # Make embedding path
            dct_path = direc.split('/')[0] + '/dct_embed/' + fam
            if not os.path.isdir(dct_path):
                os.makedirs(dct_path)

            # Load embedding and perform iDCT
            embed = np.load(emb_path)
            embed = quant2D(embed, 8, 8)  # 8x8 DCT
            np.save(f'{dct_path}/{emb}', embed)

        logging.info('iDCT performed on %s...', emb_path)


if __name__ == '__main__':
    main()
