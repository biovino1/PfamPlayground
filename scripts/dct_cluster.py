"""Compares individual DCT's from each family to its average.

__author__ = "Ben Iovino"
__date__ = "09/07/23"
"""

import os
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster


def make_matrix(transforms: list) -> pd.DataFrame:
    """Returns a pandas dataframe of DCT vectors

    :param transforms: list of DCT's for a family
    return pd.DataFrame: dataframe of all DCT's in a family
    """

    # Get the DCT vectors and domain IDs
    families = []
    features = []
    for i in range(len(transforms)):  #pylint: disable=C0200
        dct = transforms[i]
        families.append(dct[0])
        features.append(dct[1])

    # Make a dataframe and cluster
    dct_df = pd.DataFrame(features, index=families,
                        columns=[str(i) for i in range(600)])

    return dct_df


def cluster_dcts(dct_df: pd.DataFrame) -> list:
    """Returns the average dct for each cluster of DCT's.

    :param dct_df: dataframe of all DCT's in a family
    """

    # Get top 2 clusters
    link_data = linkage(dct_df, 'ward')
    num_clusters = 2
    clusters = fcluster(link_data, num_clusters, criterion='maxclust')

    # Get avg dct for each
    avg_dcts = []
    for i in range(num_clusters):
        cluster = dct_df[clusters == i + 1]
        avg_dct = cluster.mean(axis=0)
        avg_dcts.append(avg_dct)

    return avg_dcts


def get_avgs(tdirec: str, dct_db: str):
    """Saves new database of DCT's.

    :param dct_db: database of dct's
    :param tdirec: directory of transformed embeddings
    """

    dcts = []
    for fam in os.listdir(tdirec):
        transforms = np.load(f'{tdirec}/{fam}/transform.npy', allow_pickle=True)
        dct_df = make_matrix(transforms)
        avg_dcts = cluster_dcts(dct_df)

        # Add avg dcts to database
        for i, avg_dct in enumerate(avg_dcts):
            name = f'{fam}_cluster{i}'
            dcts.append(np.array([name, avg_dct], dtype=object))

    # Load database and add new dcts
    dct_db = np.load(dct_db, allow_pickle=True)
    new_db = np.concatenate((dct_db, dcts), axis=0)
    np.save('data/esm2_17_875_avg', new_db)


def main():
    """Main
    """

    tdirec = 'data/esm2_17_transform'
    dct_db = 'data/esm2_17_875_avg.npy'
    get_avgs(tdirec, dct_db)


if __name__ == '__main__':
    main()
