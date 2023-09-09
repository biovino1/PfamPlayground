"""Calculates mean, std, and range of distances between all DCT's in a family and its average.

__author__ = "Ben Iovino"
__date__ = "09/06/23"
"""

import logging
import os
import numpy as np
from scipy.spatial.distance import cityblock

log_filename = 'data/logs/dct_stats.log'  #pylint: disable=C0103
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
logging.basicConfig(filename=log_filename, filemode='w',
                     level=logging.INFO, format='%(asctime)s %(message)s')


def dct_dist(dct_db: str, tdirec: str) -> list:
    """Returns a list of average dct's and some statistics (mean, std, range) for each family.

    :param dct_db: database of dct's
    :param tdirec: directory of embeddings to transform and compare
    return list: list of average dct's and their stats
    """

    dct_db = np.load(dct_db, allow_pickle=True)
    dcts = []
    for i, fam in enumerate(os.listdir(tdirec)):

        # Get avg dct and embeds for fam
        logging.info('Calculating stats for %s %s...', fam, i)
        transforms = np.load(f'{tdirec}/{fam}/transform.npy', allow_pickle=True)
        for avg in dct_db:
            if avg[0] == fam:
                avg_dct = avg[1]
                break

        # Compare each transform to average dct and compute stats
        distances = []
        for transform in transforms:
            distances.append(1-cityblock(avg_dct, transform.trans[1]))
        mean, std = np.mean(distances), np.std(distances)
        rang = [np.min(distances), np.max(distances)]
        stats = np.array([mean, std, rang], dtype=object)

        dcts.append(np.array([fam, avg_dct, stats], dtype=object))

    return dcts


def main():
    """Main
    """

    dct_db = 'data/esm2_17_875_avg.npy'
    tdirec = 'data/esm2_17_transform'
    dcts = dct_dist(dct_db, tdirec)
    np.save('data/dct_stats', dcts)


if __name__ == '__main__':
    main()
