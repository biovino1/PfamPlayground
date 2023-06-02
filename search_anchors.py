"""================================================================================================
This script takes a query sequence and a list of anchor sequences and searches for the most
similar anchor sequence.

Ben Iovino  05/24/23   PfamPlayground
================================================================================================"""

import os


def main():

    # Call query_anchors for every query sequence in a folder
    direc = 'Data/prott5_embed'
    for fam in os.listdir(direc):
        for query in os.listdir(f'{direc}/{fam}'):

            # Call query_anchors
            print(f'{direc}/{fam}/{query}')
            os.system(f'python query_anchors.py -e {direc}/{fam}/{query}')


if __name__ == '__main__':
    main()
