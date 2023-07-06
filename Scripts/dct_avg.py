"""================================================================================================
This script takes iDCT vectors from each Pfam family and averages them to create a single vector.

Ben Iovino  07/06/23    SearchEmb
================================================================================================"""

import argparse
import numpy as np
import os
from dct_embed import quant2D
from embed_avg import get_seqs, cons_pos, get_embed, average_embed


def transform_embs(family: str, positions: dict, embeddings: dict):
    """=============================================================================================
    This function accepts a dictionary of positions that are included in the consensus sequence and
    a dictionary of embeddings. It transforms the sequence using iDCT and saves it to file.

    param family: name of Pfam family
    :param sequences: dict where seq id is key with sequence as value
    :param positions: dict where seq id is key with list of positions as value
    ============================================================================================="""

    # For each seq in embeddings, get consensus positions and transform embedding
    dcts = {}
    for seq in embeddings:
        embed = []
        for pos in positions[seq]:
            embed.append(embeddings[seq][pos])

        # Transform embedding
        embed = np.array(embed)
        dct = quant2D(embed, 3, 55)

        # Add each position in dct vec to dcts dict where position is key and list of dcts is value
        for i, pos in enumerate(dct):
            if i not in dcts:
                dcts[i] = []
            dcts[i].append(pos)

    # Average dcts and save to file
    for pos in dcts:
        dcts[pos] = round(np.mean(dcts[pos]), 0)

    # Convert dict to np array and save to file
    dcts = np.array([int(val) for val in dcts.values()])
    if not os.path.exists(f'Data/dct_avg/{family}'):
        os.makedirs(f'Data/dct_avg/{family}')
    np.save(f'Data/dct_avg/{family}/avg_dct.npy', dcts)


def main():
    """=============================================================================================
    Main
    ============================================================================================="""

    direc = 'Data/prott5_embed'
    for fam in os.listdir(direc):
        print(fam)

        # Get sequences and their consensus positions
        sequences = get_seqs(fam)
        positions = cons_pos(sequences)

        # Get embeddings for each sequence in family and average them
        embed_direc = f'{direc}/{fam}'
        embeddings = get_embed(embed_direc, sequences)
        transform_embs(fam, positions, embeddings)


if __name__ == '__main__':
    main()
