"""================================================================================================
This script takes DCT transforms from npy files and combines them into a single numpy array.

Ben Iovino  07/24/23   SearchEmb
================================================================================================"""

import os
import numpy as np
from util import Transform


def combine(files: list):
    """=============================================================================================
    This function takes a list of npy files containing DCT transforms and combines them into a
    single transform.

    :param files: list of npy files
    ============================================================================================="""

    # Get all transforms
    trdict = {}
    for file in files:
        with open(file, 'rb') as emb:
            trans = np.load(emb, allow_pickle=True)
        for tran in trans:
            trdict[tran[0]] = trdict.get(tran[0], []) + [tran[1]]

    # Concatenate DCTs
    dcts = []
    for fam, trans in trdict.items():
        dct = Transform(fam, None, None)

        # If less transforms than files, skip
        if len(trans) < len(files):
            continue
        for tran in trans:
            dct.concat(tran)
        dcts.append(dct.trans)

    # Save combined transforms
    with open('data/comb_dct.npy', 'wb') as emb:
        np.save(emb, dcts)


def main():
    """=============================================================================================
    Main calls combine() to combine all DCT transforms from the list of files.
    ============================================================================================="""

    # Get all avg transform files from data directory
    avg_files = []
    for file in os.listdir('data'):
        if file.endswith('avg.npy'):
            avg_files.append(f'data/{file}')

    # Sort files by number of layers
    avg_files = sorted(avg_files, key=lambda x: int(x.split('_')[1]))
    combine(avg_files)


if __name__ == '__main__':
    main()
