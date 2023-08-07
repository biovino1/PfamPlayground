"""This script finds the average iDCT of embeddings for each Pfam family and saves them to file.

__author__ = "Ben Iovino"
__date__ = "07/06/23"
"""

import argparse
import os
import logging
import numpy as np
from util import Transform
from embed_avg import get_seqs, cons_pos, get_embed

log_filename = 'data/logs/dct_avg.log'  #pylint: disable=C0103
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
logging.basicConfig(filename=log_filename, filemode='w',
                     level=logging.INFO, format='%(message)s')


def transform_avgs(
        fam: str, positions: dict, embeddings: dict, args: argparse.Namespace) -> np.ndarray:
    """Averages the embeddings for each position in a dictionary of embeddings and returns the
    iDCT vector of the averaged embedding as a numpy array.

    :param positions: dict where seq id is key with list of positions as value
    :param embeddings: dict where seq is key with list of embeddings as value
    :param args: argparse.Namespace object with dct dimensions
    :return: numpy array of iDCT vector
    """

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
    avg_embed = Transform(fam, np.array(avg_embed), None)
    avg_embed.quant_2D(args.s1, args.s2)

    return avg_embed


def main():
    """Main goes through each Pfam family and calls get_seqs() to get protein sequences, cons_pos()
    to get the consensus sequence positions, get_embed() to get the embeddings for each sequence,
    one of the transform functions to transform the embeddings. All of the transformed embeddings
    are saved to a numpy array.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='data/esm2_17_embed', help='direc of embeds to avg')
    parser.add_argument('-s1', type=int, default=6)
    parser.add_argument('-s2', type=int, default=50)
    args = parser.parse_args()

    dcts = []
    for i, fam in enumerate(os.listdir(args.d)):
        logging.info('Averaging embeddings for %s, %s', fam, i)

        # Get sequences and their consensus positions
        sequences = get_seqs(fam)
        positions = cons_pos(sequences)

        # Get embeddings for each sequence in family and average them
        embed_direc = f'{args.d}/{fam}'
        embeddings = get_embed(embed_direc, sequences)

        # Transform each average embedding and store in list
        avg_dct = transform_avgs(fam, positions, embeddings, args)
        if avg_dct.trans[1] is not None:
            dcts.append(avg_dct.trans)

    # Save all dcts to file
    enclay = '_'.join(args.d.split('/')[-1].split('_')[:2])  # enc/layer used to embed
    np.save(f'data/{enclay}_{args.s1}{args.s2}_avg.npy', dcts)


if __name__ == '__main__':
    main()
