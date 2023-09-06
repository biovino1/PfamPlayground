"""Calculates mean, std, and range of distances between all DCT's in a family and its average.

__author__ = "Ben Iovino"
__date__ = "09/06/23"
"""

import os
import numpy as np
from scipy.spatial.distance import cityblock
from util import Transform


def dct_dist(dct_db: str, edirec: str) -> list:
    """Returns a list of average dct's and some statistics (mean, std, range) for each family.

    :param dct_db: database of dct's
    :param edirec: directory of embeddings to transform and compare
    return list: list of average dct's and their stats
    """

    dct_db = np.load(dct_db, allow_pickle=True)
    dcts = []
    for fam in os.listdir(edirec):

        # Get avg dct and embeds for fam
        embeds = np.load(f'{edirec}/{fam}/embed.npy', allow_pickle=True)
        for avg in dct_db:
            if avg[0] == fam:
                avg_dct = avg[1]
                break

        # Transform each embed in directory and store in list
        transforms = []
        for embed in embeds:
            transform = Transform(embed[0], embed[1], None)
            transform.quant_2D(8, 75)
            transforms.append(transform)

        # Compare each transform to average dct and compute stats
        distances = []
        for transform in transforms:
            distances.append(1-cityblock(avg_dct, transform.trans[1]))
        mean, std = np.mean(distances), np.std(distances)
        rang = range(np.min(distances), np.max(distances))
        stats = np.array([mean, std, rang], dtype=object)

        dcts.append(np.array([fam, avg_dct, stats], dtype=object))

    return dcts


def main():
    """Main
    """

    dct_db = 'data/esm2_17_875_avg.npy'
    edirec = 'data/esm2_17_embed'
    dcts = dct_dist(dct_db, edirec)
    np.save('data/dct_stats', dcts)


if __name__ == '__main__':
    main()
