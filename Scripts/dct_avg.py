"""================================================================================================
This script takes iDCT vectors from each Pfam family and averages them to create a single vector.

Ben Iovino  07/06/23    SearchEmb
================================================================================================"""

import numpy as np
import os
from dct_embed import quant2D
from embed_avg import get_seqs, cons_pos, get_embed


def transform_cons(family: str, positions: dict, embeddings: dict):
    """=============================================================================================
    This function accepts a dictionary of positions that are included in the consensus sequence and
    a dictionary of embeddings. It transforms the sequence with only consensus positions using iDCT
    and saves it to file.

    :param family: name of Pfam family
    :param positions: dict where seq id is key with list of positions as value
    :param embeddings: dict where seq is key with list of embeddings as value
    ============================================================================================="""

    # For each seq in embeddings, get consensus positions and transform embedding
    dcts = {}
    for seq in embeddings:
        embed = []
        for pos in positions[seq]:
            embed.append(embeddings[seq][pos])

        # Transform embedding
        embed = np.array(embed)
        dct = quant2D(embed, 5, 44)

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


def transform_embs(family: str, embeddings: dict):
    """=============================================================================================
    This function accepts a dictionary of embeddings and transforms them using iDCT. It then saves
    them to file.

    :param family: name of Pfam family
    :param embeddings: dict where seq is key with list of embeddings as value
    ============================================================================================="""

    dcts = []
    for seq in embeddings:
        embed = []
        for pos in embeddings[seq]:
            if isinstance(pos, int) is False:
                embed.append(pos)
        embed = np.array(embed)
        dcts.append(quant2D(embed, 5, 44))

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

    direc = 'Data/prott5_embed'
    for fam in os.listdir(direc):

        # Get sequences and their consensus positions
        sequences = get_seqs(fam)
        positions = cons_pos(sequences)

        # Get embeddings for each sequence in family and average them
        embed_direc = f'{direc}/{fam}'
        embeddings = get_embed(embed_direc, sequences)
        transform_cons(fam, positions, embeddings)
        transform_embs(fam, embeddings)


if __name__ == '__main__':
    main()
