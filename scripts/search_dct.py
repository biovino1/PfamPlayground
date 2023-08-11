"""This script takes a query sequence and a database of dct vectors and outputs the most similar dct
vectors along with their similarity scores.

__author__ = "Ben Iovino"
__date__ = "07/05/23"
"""

import argparse
import datetime
import logging
import os
import pickle
from random import sample
from Bio import SeqIO
import numpy as np
import torch
from util import load_model, Embedding, Transform
from scipy.spatial.distance import cityblock

log_filename = 'data/logs/search_dct.log'  #pylint: disable=C0103
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
logging.basicConfig(filename=log_filename, filemode='w',
                     level=logging.INFO, format='%(message)s')


def embed_query(
    sequence: str, tokenizer, model, device: str, encoder: str, layer: int) -> np.ndarray:
    """Returns the embedding of a fasta sequence.

    :param sequence: path to fasta file containing query sequence
    :param tokenizer: tokenizer
    :param model: encoder model
    :param device: cpu or gpu
    :param encoder: prott5 or esm2
    :param layer: layer to extract features from (if using esm2)
    :return: embedding of query sequence
    """

    # Get seq from file
    seq = ()
    with open(sequence, 'r', encoding='utf8') as f:
        for seq in SeqIO.parse(f, 'fasta'):
            seq = (seq.id, str(seq.seq))

    # Initialize Embedding object and embed sequence
    embed = Embedding(seq[0], seq[1], None)
    embed.embed_seq(tokenizer, model, device, encoder, layer)

    return embed.embed[1]


def get_transform(seq: str, tokenizer, model, device: str, args: argparse.Namespace) -> Transform:
    """Returns the DCT of an embedded fasta sequence.

    :param seq: path to fasta file containing query sequence
    :param tokenizer: tokenizer
    :param model: encoder model
    :param device: cpu or gpu
    :param args: argparse.Namespace object containing arguments
    :return: Transform object containing dct representation of query sequence
    """

    # Get DCT for each layer
    query = '/'.join(seq.split('/')[2:])
    trans = {}
    for i, layer in enumerate(args.l):
        embed = embed_query(seq, tokenizer, model, device, args.e, layer)
        embed = Transform(query, np.array(embed), None)
        embed.quant_2D(args.s1[i], args.s2[i])
        if embed.trans[1] is None:  # Skip if DCT is None
            return None  #\\NOSONAR
        trans[layer] = embed

    # Concatenate DCTs
    dct = Transform(query, None, None)
    for layer, tran in trans.items():
        dct.concat(tran.trans[1])

    return dct


def query_sim(dct: np.ndarray, query: np.ndarray, sims: dict, fam: str, metric: str) -> dict:
    """Finds the similarity between two arrays and returns a dictionary of similarities between
    database and query vectors.

    :param dct: dct vector of sequence from database
    :param query: dct vector of embedded query sequence
    :param sims: dictionary of similarities between dct and query vectors
    :param fam: name of dct sequence
    :param metric: similarity metric
    :return: updated dict of similarities between dct and query vectors
    """

    # Cosine similarity between dct and query embedding
    if metric == 'cosine':
        sim = np.dot(dct, query) / \
            (np.linalg.norm(dct) * np.linalg.norm(query))
        sims[fam].append(sim)

    # City block distance between dct and query embedding
    if metric == 'cityblock':
        dist = 1/cityblock(dct, query)
        sims[fam].append(dist)

    return sims


def query_search(query: np.ndarray, search_db: np.ndarray, results: int, metric: str) -> dict:
    """Returns the top n results from searching a query sequence against a database of dct vectors.

    :param query: embedding of query sequence
    :param search_db: np array of dct embeddings
    :param results: number of results to return
    :param metric: similarity metric

    :return: dict where keys are dct embeddings and values are similarity scores
    """

    # Search query against every dct embedding
    sims = {}
    for dct in search_db:
        fam, dct = dct[0], dct[1]  # family name, dct vector for family
        if fam not in sims:  # store sims in dict
            sims[fam] = []
        sims = query_sim(dct, query, sims, fam, metric)  # compare query to dct

    # Sort sims dict and return top n results
    top_sims = {}
    for key in sorted(sims, key=sims.get, reverse=True)[:results]:
        top_sims[key] = sims[key]

    return top_sims


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


def search_results(query: str, results: dict) -> dict:
    """Returns a dict of counts for matches, top n results, and same clan for all queries in a
    search.

    :param query: query sequence
    :param results: dictionary of results from searching query against dcts
    :return: dict of counts for matches, top n results, and same clan
    """

    # Log time and similarity for top 5 results
    logging.info('%s\n%s', datetime.datetime.now(), query)
    for fam, sim in list(results.items())[:5]:
        logging.info('%s,%s', fam, sim)

    # See if query is in top results
    results_fams = [fam.split('/')[0] for fam in results.keys()]
    query_fam = query.split('/')[0]
    counts = {'match': 0, 'top': 0, 'clan': 0}
    if query_fam == results_fams[0]:  # Top result
        counts['match'] += 1
        return counts
    if query_fam in results_fams:  # Top n results
        counts['top'] += 1
        return counts
    counts['clan'] += clan_results(query_fam, results_fams)  # Same clan

    return counts


def main():
    """Main function loads tokenizer and model, randomly samples a query sequence from a family,
    embeds and transforms the query, searches the query against DCT vectors, and logs the results
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='data/full_dct/full_dct.npy')
    parser.add_argument('-e', type=str, default='esm2')
    parser.add_argument('-l', type=int, nargs='+', default=[17])
    parser.add_argument('-t', type=int, default=100)
    parser.add_argument('-s1', type=int, nargs='+', default=[8])
    parser.add_argument('-s2', type=int, nargs='+', default=[80])
    args = parser.parse_args()

    # Load tokenizer and encoder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # pylint: disable=E1101
    tokenizer, model = load_model(args.e, device)

    # Load embed/dct database
    search_db = np.load(args.d, allow_pickle=True)
    search_fams = [transform[0] for transform in search_db]

    # Call query_search for every query sequence in a folder
    counts = {'match': 0, 'top': 0, 'clan': 0, 'total': 0}
    direc = 'data/full_seqs'
    for fam in os.listdir(direc):
        if fam not in search_fams:
            continue

        # Randomly sample one query from family and get it appropriate dct
        queries = os.listdir(f'{direc}/{fam}')
        query = sample(queries, 1)[0]
        seq_file = f'{direc}/{fam}/{query}'
        query = get_transform(seq_file, tokenizer, model, device, args)
        if query is None:
            logging.info('%s\n%s\nQuery was too small for transformation dimensions',
                          datetime.datetime.now(), query)
            continue

        # Search idct embeddings and analyze results
        results = query_search(query.trans[1], search_db, args.t, 'cityblock')
        search_counts = search_results(query.trans[0], results)
        counts['match'] += search_counts['match']
        counts['top'] += search_counts['top']
        counts['clan'] += search_counts['clan']
        counts['total'] += 1
        logging.info('Queries: %s, Matches: %s, Top%s: %s, Clan: %s\n',
                      counts['total'], counts['match'], args.t, counts['top'], counts['clan'])


if __name__ == '__main__':
    main()
