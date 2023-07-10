"""================================================================================================
This script takes iDCT vectors from each Pfam family and averages them to create a single vector.

Ben Iovino  07/06/23    SearchEmb
================================================================================================"""

import argparse
import os
import numpy as np
from dct_embed import quant2D
from embed_avg import get_seqs, cons_pos, get_embed


def transform_cons(family: str, positions: dict, embeddings: dict, args: argparse.Namespace):
    """=============================================================================================
    This function accepts a dictionary of positions that are included in the consensus sequence and
    a dictionary of embeddings. It transforms the sequence with only consensus positions using iDCT
    and saves it to file.

    :param family: name of Pfam family
    :param positions: dict where seq id is key with list of positions as value
    :param embeddings: dict where seq is key with list of embeddings as value
    :param args: argparse.Namespace object with dct dimensions
    ============================================================================================="""

    # For each seq in embeddings, get consensus positions and transform embedding
    dcts = {}
    for seq in embeddings:
        embed = []
        for pos in positions[seq]:
            embed.append(embeddings[seq][pos])

        # Transform embedding unless dimensions are too small
        embed = np.array(embed)
        try:
            dct = quant2D(embed, args.s1, args.s2)
        except ValueError:
            continue

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
    if not os.path.exists(f'Data/avg_dct/{family}'):
        os.makedirs(f'Data/avg_dct/{family}')
    np.save(f'Data/avg_dct/{family}/avg_dct.npy', dcts)


def transform_embs(family: str, embeddings: dict, args: argparse.Namespace):
    """=============================================================================================
    This function accepts a dictionary of embeddings and transforms them using iDCT. It then saves
    them to file.

    :param family: name of Pfam family
    :param embeddings: dict where seq is key with list of embeddings as value
    :param args: argparse.Namespace object with dct dimensions
    ============================================================================================="""

    dcts = []
    for seq in embeddings:
        embed = []
        for pos in embeddings[seq]:
            if isinstance(pos, int) is False:  # Ignore 0's that were added for padding
                embed.append(pos)

        # Transform embedding unless dimensions are too small
        embed = np.array(embed)
        try:
            dcts.append(quant2D(embed, args.s1, args.s2))
        except ValueError:
            continue

    # Average dcts and save to file
    dcts = np.mean(dcts, axis=0)

    # Convert dict to np array and save to file
    dcts = np.array([int(val) for val in dcts])
    if not os.path.exists(f'Data/avg_dct/{family}'):
        os.makedirs(f'Data/avg_dct/{family}')
    np.save(f'Data/avg_dct/{family}/avg_dct.npy', dcts)


def main():
    """=============================================================================================
    Main goes through each Pfam family and calls get_seqs() to get protein sequences, cons_pos() to
    get the consensus sequence positions, get_embed() to get the embeddings for each sequence, and
    either transform_cons to get DCT transform of embeddings including only the consensus positions
    or transform_embs to get DCT transform of entire embeddings and save them to file.
    ============================================================================================="""

    parser = argparse.ArgumentParser()
    parser.add_argument('-s1', type=int, default=5)
    parser.add_argument('-s2', type=int, default=44)
    parser.add_argument('-t', type=str, default='cons')
    args = parser.parse_args()

    direc = 'Data/prott5_embed'
    for fam in os.listdir(direc):

        # Check if average embedding already exists
        if os.path.exists(f'Data/avg_dct/{fam}/avg_dct.npy'):
            continue

        # Get sequences and their consensus positions
        sequences = get_seqs(fam)
        positions = cons_pos(sequences)

        # Get embeddings for each sequence in family and average them
        embed_direc = f'{direc}/{fam}'
        embeddings = get_embed(embed_direc, sequences)
        if args.t == 'cons':
            transform_cons(fam, positions, embeddings, args)
        if args.t == 'all':
            transform_embs(fam, embeddings, args)


if __name__ == '__main__':
    main()
