"""================================================================================================
This script accepts a directory of embeddings and returns their inverse discrete cosine transforms.

Ben Iovino  07/04/23   SearchEmb
================================================================================================"""

import argparse
import os
import logging
import numpy as np
from scipy.fft import dct, idct


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

    logging.basicConfig(filename='Data/dct_embed.log',
                     level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='Data/prott5_embed')
    parser.add_argument('-s1', type=int, default=5)
    parser.add_argument('-s2', type=int, default=44)
    args = parser.parse_args()

    # Go through embeddings and call idct on each one
    for fam in os.listdir(args.d):  # Each family in the directory
        logging.info('iDCT being performed on %s...', fam)

        # Skip family if iDCT already performed
        dct_direc = '/'.join(args.d.split("/")[:-1])  # Take off last directory
        dct_direc = f'{dct_direc}/dct_embed'
        if os.path.isdir(f'{dct_direc}/{fam}'):
            logging.info('iDCT already performed on %s\n', fam)
            continue

        for emb in os.listdir(f'{args.d}/{fam}'):  # Each emb in the family
            logging.info('iDCT being performed on %s...', emb)
            emb_path = f'{args.d}/{fam}/{emb}'

            # Make embedding path
            dct_path = f'{dct_direc}/{fam}'
            if not os.path.isdir(dct_path):
                os.makedirs(dct_path)

            # Load embedding and perform iDCT
            embed = np.load(emb_path)
            embed = quant2D(embed, args.s1, args.s2)  # nxn 1D array
            np.save(f'{dct_path}/{emb}', embed)
        logging.info('iDCT performed on %s\n', fam)


if __name__ == '__main__':
    main()
