"""================================================================================================
This script averages all embeddings for each Pfam family and saves the average embedding as a numpy
array in a new directory.

Ben Iovino  05/12/23   PfamPlayground
================================================================================================"""

import os
import numpy as np

def main():

    for family in os.listdir('prott5_embed'):

        # Get embeddings for each sequence in family
        embeddings = {}
        for file in os.listdir(f'prott5_embed/{family}'):
            with open(f'prott5_embed/{family}/{file}', 'r', encoding='utf8') as file:
                embeddings[file] = np.loadtxt(file)

        for key, item in embeddings.items():
            print(key, len(item))


if __name__ == '__main__':
    main()
