"""This script searches two different databases, first using dct vectors to filter out
dissimilar sequences, then using embeddings to find the most similar sequences.

__author__ = "Ben Iovino"
__date__ = "08/18/23"
"""

import argparse
import datetime
import logging
import os
import pickle
from random import sample
import numpy as np
from Bio import SeqIO
import torch
from util import load_model, Embedding, Transform

log_filename = 'data/logs/search.log'  #pylint: disable=C0103
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
logging.basicConfig(filename=log_filename, filemode='w',
                     level=logging.INFO, format='%(message)s')


def embed_query(
    fam: str, tokenizer, model, device: str, args: argparse.Namespace) -> tuple:
    """Returns the embedding of a fasta sequence.

    :param fam: family of query sequence
    :param tokenizer: tokenizer
    :param model: encoder model
    :param device: cpu or gpu
    :param args: command line arguments
    :return: Embedding and Transform objects
    """

    # Get seqs from file and randomly sample one
    seqs = {}
    with open(f'data/full_seqs/{fam}/seqs.fa', 'r', encoding='utf8') as f:
        for i, seq in enumerate(SeqIO.parse(f, 'fasta')):
            seqs[i] = (seq.id, str(seq.seq))
    seq = sample(list(seqs.values()), 1)[0]

    # Initialize Embedding object and embed sequence
    embed = Embedding(seq[0], seq[1], None)
    embed.embed_seq(tokenizer, model, device, args.e, args.l)

    # DCT embedding
    transform = Transform(embed.embed[0], embed.embed[1], None)
    transform.quant_2D(args.s1, args.s2)

    return embed, transform


def get_fams(results: dict) -> list:
    """Returns a list of family names from a dictionary of search results:

    :param results: dict where key is family name and value is similarity score
    :return: list of family names
    """

    result_fams = []
    for name in results.keys():
        if '_cluster' in name:
            result_fams.append('_'.join(name.split('_')[:-1]))
        else:
            result_fams.append(name)

    return result_fams


def clan_results(query_fam: str, results_fams: list) -> int:
    """Returns 1 if query and top result are in the same clan, 0 otherwise.

    :param query_fam: family of query sequence
    :param results_fams: list of families of top N results
    :return: 1 if query and top result are in the same clan, 0 otherwise
    """

    with open('data/clans.pkl', 'rb') as file:
        clans = pickle.load(file)
    for fams in clans.values():
        if query_fam in fams and results_fams[0] in fams:
            return 1
    return 0


def search_results(query: str, results: dict, counts: dict) -> dict:
    """Returns a dict of counts for matches, top n results, and same clan for all queries in a
    search.

    :param query: query sequence
    :param results: dictionary of results from searching query against dcts
    :param counts: dictionary of counts for matches, top n results, and same clan
    :param top: number of results to return
    :return: dict of counts for matches, top n results, and same clan
    """

    # Log time and similarity for top 5 results
    logging.info('%s\n%s', datetime.datetime.now(), query)
    for fam, sim in list(results.items())[:5]:
        logging.info('%s,%s', fam, sim)

    # See if query is in top results
    results_fams = get_fams(results)
    query_fam = query.split('/')[0]
    counts['total'] += 1
    if query_fam == results_fams[0]:  # Top result
        counts['match'] += 1
        return counts
    if query_fam in results_fams:  # Top n results
        counts['top'] += 1
        return counts
    counts['clan'] += clan_results(query_fam, results_fams)  # Same clan

    return counts


def main():
    """Searches two different databases, first using dct vectors to filter out dissimilar sequences.
    If top result is not same as query family, then searches embeddings database.

    args:
        -dct: database of dct vectors
        -emb: database of embeddings (leave empty if only searching dct)
        -e: encoder model
        -l: layer of model to use (for esm2 only)
        -t: number of results to return from search
        -s1: first dimension of dct
        -s2: second dimension of dct
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-dct', type=str, default='data/esm2_17_875_clusters.npy')
    parser.add_argument('-emb', type=str, default='')
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
    if args.emb != '':
        emb_db = np.load(args.emb, allow_pickle=True)

    # Call query_search for every query sequence in a folder
    counts = {'match': 0, 'top': 0, 'clan': 0, 'total': 0}
    for fam in os.listdir('data/full_seqs'):

        # Get random sequence from family and embed/transform sequence
        embed, dct = embed_query(fam, tokenizer, model, device, args)
        if dct.trans[1] is None:
            logging.info('%s\n%s\nQuery was too small for transformation dimensions',
                          datetime.datetime.now(), embed.embed[0])
            continue

        # Search dct db - check if top family is same as query family
        results = dct.search(dct_db, args.t)
        results_fams = get_fams(results)
        if fam == results_fams[0] or args.emb == '':
            counts = search_results(f'{fam}/{dct.trans[0]}', results, counts)
            logging.info('DCT: Queries: %s, Matches: %s, Top%s: %s, Clan: %s\n',
                        counts['total'], counts['match'], args.t, counts['top'], counts['clan'])
            continue

        # If top family is not same as query family, search anchors on top results from DCTs
        results = embed.search(emb_db, args.t, results_fams)
        counts = search_results(f'{fam}/{embed.embed[0]}', results, counts)
        logging.info('ANCHORS: Queries: %s, Matches: %s, Top%s: %s, Clan: %s\n',
                      counts['total'], counts['match'], args.t, counts['top'], counts['clan'])


if __name__ == '__main__':
    main()
