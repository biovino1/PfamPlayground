"""This script searches two different databases, first using dct vectors to filter out
dissimilar sequences, then using embeddings to find the most similar sequences.

__author__ = "Ben Iovino"
__date__ = "08/18/23"
"""


import argparse
import datetime
import logging
import os
from random import sample
import numpy as np
import torch
from search_anchors import embed_query, search_results
from util import load_model, Embedding, Transform

log_filename = 'data/logs/search.log'  #pylint: disable=C0103
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
logging.basicConfig(filename=log_filename, filemode='w',
                     level=logging.INFO, format='%(message)s')


def transform_embed(embed: Embedding, args: argparse.Namespace) -> Transform:
    """Returns the DCT of an embedded fasta sequence.

    :param embed: Embedding object
    :param args: argparse.Namespace object containing arguments
    :return: Transform object containing dct representation of query sequence
    """

    transform = Transform(embed.embed[0], embed.embed[1], None)
    transform.quant_2D(args.s1, args.s2)
    if transform.trans[1] is None:  # Skip if DCT is None
        return None  #\\NOSONAR

    return transform


def main():
    """Main
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-dct', type=str, default='data/esm2_17_875_avg.npy')
    parser.add_argument('-emb', type=str, default='data/anchors.npy')
    parser.add_argument('-e', type=str, default='esm2')
    parser.add_argument('-l', type=int, default=17)
    parser.add_argument('-t', type=int, default=100)
    parser.add_argument('-s1', type=int, default=8)
    parser.add_argument('-s2', type=int, default=75)
    args = parser.parse_args()

    # Load tokenizer and encoder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # pylint: disable=E1101
    tokenizer, model = load_model(args.e, device)

    # Load embed/dct database
    dct_db = np.load(args.dct, allow_pickle=True)
    dct_fams = [transform[0] for transform in dct_db]
    emb_db = np.load(args.emb, allow_pickle=True)

    # Call query_search for every query sequence in a folder
    counts = {'match': 0, 'top': 0, 'clan': 0, 'total': 0}
    direc = 'data/full_seqs'
    for fam in dct_fams:

        # Randomly sample one query from family and get it appropriate dct
        queries = os.listdir(f'{direc}/{fam}')
        query = sample(queries, 1)[0]
        seq_file = f'{direc}/{fam}/{query}'
        embed = embed_query(seq_file, tokenizer, model, device, args)
        dct = transform_embed(embed, args)
        if dct is None:
            logging.info('%s\n%s\nQuery was too small for transformation dimensions',
                          datetime.datetime.now(), query)
            continue

        # Search idct embeddings, take top families, and search embeddings
        results = dct.search(dct_db, args.t)
        results_fams = [fam.split('/')[0] for fam in results.keys()]
        results = embed.search(emb_db, args.t, results_fams)
        counts = search_results(f'{fam}/{query}', results, counts)
        logging.info('Queries: %s, Matches: %s, Top%s: %s, Clan: %s\n',
                      counts['total'], counts['match'], args.t, counts['top'], counts['clan'])
        break


if __name__ == '__main__':
    main()
