"""================================================================================================
This script finds the average iDCT of embeddings for each Pfam family and saves them to file.

Ben Iovino  07/06/23    SearchEmb
================================================================================================"""

import argparse
import os
import numpy as np
from dct_embed import quant2D
from embed_avg import get_seqs, cons_pos, get_embed


def transform_avgs(positions: dict, embeddings: dict, args: argparse.Namespace) -> np.ndarray:
    """=============================================================================================
    This function accepts a dictionary of positions that are included in the consensus sequence and
    a dictionary of embeddings. It averages the embeddings for each position and returns the iDCT
    vector of the whole embedding as a numpy array.

    :param sequences: dict where seq id is key with sequence as value
    :param positions: dict where seq id is key with list of positions as value
    :param embeddings: dict where seq is key with list of embeddings as value
    :param args: argparse.Namespace object with dct dimensions
    :return avg_dct: numpy array of iDCT vector
    ============================================================================================="""

    # Create a dict of lists where each list contains the embeddings for a position in the consensus
    seq_embed = {}
    for seqid, position in positions.items():  # Go through positions for each sequence
        for pos in position:
            if pos not in seq_embed:  # Initialize key and value
                seq_embed[pos] = []
            seq_embed[pos].append(embeddings[seqid][pos])  # Add embedding to list for that pos

    # Sort dictionary by key (out of order for some reason)
    seq_embed = dict(sorted(seq_embed.items()))

    # Move through each position (list of embeddings) and average them
    avg_embed = []
    for pos, embed in seq_embed.items():
        avg_embed.append(np.mean(embed, axis=0))  # Find mean for each position (float)

    # Perform idct on avg_embed
    avg_embed = np.array(avg_embed)
    avg_dct = quant2D(avg_embed, args.s1, args.s2)  # nxn 1D array

    return avg_dct


def transform_cons(positions: dict, embeddings: dict, args: argparse.Namespace) -> np.ndarray:
    """=============================================================================================
    This function accepts a dictionary of positions that are included in the consensus sequence and
    a dictionary of embeddings. It transforms the sequence with only consensus positions using iDCT
    and returns the iDCT vector as a numpy array.

    :param positions: dict where seq id is key with list of positions as value
    :param embeddings: dict where seq is key with list of embeddings as value
    :param args: argparse.Namespace object with dct dimensions
    :return avg_dct: numpy array of iDCT vector
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

    # Average dcts for each position and convert to np array
    for pos in dcts:
        dcts[pos] = round(np.mean(dcts[pos]), 0)
    avg_dct = np.array([int(val) for val in dcts.values()])

    return avg_dct


def transform_embs(embeddings: dict, args: argparse.Namespace) -> np.ndarray:
    """=============================================================================================
    This function accepts a dictionary of embeddings, transforms each one with iDCT, and averages
    them. It returns the iDCT vector as a numpy array.

    :param embeddings: dict where seq is key with list of embeddings as value
    :param args: argparse.Namespace object with dct dimensions
    :return avg_dct: numpy array of iDCT vector
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

    # Average dcts and convert to np array
    dcts = np.mean(dcts, axis=0)
    avg_dct = np.array([int(val) for val in dcts])

    return avg_dct


def main():
    """=============================================================================================
    Main goes through each Pfam family and calls get_seqs() to get protein sequences, cons_pos() to
    get the consensus sequence positions, get_embed() to get the embeddings for each sequence, and
    either transform_cons to get DCT transform of embeddings including only the consensus positions
    or transform_embs to get DCT transform of entire embeddings and save them to file.
    ============================================================================================="""

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='Data/prott5_embed', help='direc of embeds to avg')
    parser.add_argument('-s1', type=int, default=5)
    parser.add_argument('-s2', type=int, default=44)
    parser.add_argument('-t', type=str, default='avg', help ='avg, cons, or all')
    args = parser.parse_args()

    dcts = []
    for fam in os.listdir(args.d):

        # Get sequences and their consensus positions
        sequences = get_seqs(fam)
        positions = cons_pos(sequences)

        # Get embeddings for each sequence in family and average them
        embed_direc = f'{args.d}/{fam}'
        embeddings = get_embed(embed_direc, sequences)
        if args.t == 'avg':
            avg_dct = transform_avgs(positions, embeddings, args)
        if args.t == 'cons':
            avg_dct = transform_cons(positions, embeddings, args)
        if args.t == 'all':
            avg_dct = transform_embs(embeddings, args)

        # Store fam and its avg_dct in a numpy array
        test = np.array([fam, avg_dct], dtype=object)
        dcts.append(test)

    np.save('Data/avg_dct.npy', dcts)


if __name__ == '__main__':
    main()
