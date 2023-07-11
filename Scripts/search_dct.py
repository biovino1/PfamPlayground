"""================================================================================================
This script takes a query sequence and a database of dct vectors and outputs the most similar dct
vectors along with their similarity scores.

Ben Iovino  07/05/23   SearchEmb
================================================================================================"""

import argparse
import datetime
import logging
import os
import pickle
from random import sample
import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer
from utility import prot_t5xl_embed, load_model
from dct_embed import quant2D
from scipy.spatial.distance import cityblock

logging.basicConfig(filename='Data/dct_search.log',
                     level=logging.INFO, format='%(message)s')


def embed_query(sequence: str, tokenizer: T5Tokenizer, model: T5EncoderModel, device) -> np.ndarray:
    """=============================================================================================
    This function embeds a query sequence and returns it.

    :param sequence: path to fasta file containing query sequence
    :param tokenizer: T5 tokenizer
    :param model: ProtT5-XL model
    :param device: cpu or gpu
    :return np.ndarray: embedding of query sequence
    ============================================================================================="""

    # Embed query in this script to save time from loading model every time
    with open(sequence, 'r', encoding='utf8') as fa_file:
        seq = ''.join([line.strip('\n') for line in fa_file.readlines()[1:]])
    embed = prot_t5xl_embed(seq, tokenizer, model, device)

    return embed


def query_sim(dct: np.ndarray, query: np.ndarray, sims: dict, seq: str, metric: str) -> dict:
    """=============================================================================================
    This function takes a dct vector (1D array) and a query dct vector (1D array) and finds the
    similarity between the two vectors

    :param dct: dct vector of sequence from database
    :param query: dct vector of embedded query sequence
    :param sims: dictionary of similarities between dct and query vectors
    :param family: name of dct sequence
    :param metric: similarity metric
    :return float: updated dictionary
    ============================================================================================="""

    # Cosine similarity between dct and query embedding
    if metric == 'cosine':
        sim = np.dot(dct, query) / \
            (np.linalg.norm(dct) * np.linalg.norm(query))
        sims[seq].append(sim)

    # City block distance between dct and query embedding
    if metric == 'cityblock':
        dist = 1/cityblock(dct, query)
        sims[seq].append(dist)

    return sims


def query_search(query: np.ndarray, dcts: str, results: int, metric: str) -> dict:
    """=============================================================================================
    This function takes a query embedding, a directory of dct embeddings, and returns a dict
    of the top n most similar dct embeddings.

    :param query: embedding of query sequence
    :param dcts: directory of dcts embeddings
    :param results: number of results to return
    :param metric: similarity metric
    :return top_sims: dict where keys are dct embeddings and values are similarity scores
    ============================================================================================="""

    # Search query against every dct embedding
    sims = {}
    for family in os.listdir(dcts):
        for dct in os.listdir(f'{dcts}/{family}'):
            dct_emb = np.load(f'{dcts}/{family}/{dct}')

            # Add seq to sims dict
            seq = f'{family}/{dct.strip(".npy")}'
            if seq not in sims:
                sims[seq] = []

            # Compare every dct embedding to query
            sims = query_sim(dct_emb, query, sims, seq, metric)

    # Sort sims dict and return top n results
    top_sims = {}
    for key in sorted(sims, key=sims.get, reverse=True)[:results]:
        top_sims[key] = sims[key]

    return top_sims


def search_results(query: str, results: dict):
    """=============================================================================================
    This function compares a query sequence to a dictionary of results.

    :param query: query sequence
    :param results: dictionary of results from searching query against dcts
    ============================================================================================="""

    # Log time and similarity for each result
    logging.info('%s\n%s', datetime.datetime.now(), query)
    for fam, sim in results.items():
        logging.info('%s,%s', fam, sim)

    # See if query is in top results
    results_fams = [fam.split('/')[0] for fam in results.keys()]
    query_fam = query.split('/')[0]
    match, top, clan = 0, 0, 0
    if query_fam == results_fams[0]:  # Top result
        match += 1
        return match, top, clan
    if query_fam in results_fams:  # Top n results
        top += 1
        return match, top, clan

    # Read clans dict and see if query is in same clan as top result
    with open('Data/clans.pkl', 'rb') as file:
        clans = pickle.load(file)
    for fams in clans.values():
        if query_fam in fams and results_fams[0] in fams:
            clan += 1
            return match, top, clan

    return match, top, clan


def main():
    """=============================================================================================
    Main function loads tokenizer and model, randomly samples a query sequence from a family, embeds
    and transforms the query, searches the query against DCT vectors, and logs the results
    ============================================================================================="""

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='Data/avg_dct', help='direc of embeds to search')
    args = parser.parse_args()

    # Load tokenizer and encoder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # pylint: disable=E1101
    tokenizer, model = load_model('prott5', device)

    # Call query_search for every query sequence in a folder
    match, top, clan, total = 0, 0, 0, 0
    direc = 'Data/full_seqs'
    for fam in os.listdir(direc):
        if fam not in os.listdir(args.d):
            continue

        # Randomly sample one query from family
        queries = os.listdir(f'{direc}/{fam}')
        query = sample(queries, 1)[0]
        seq_file = f'{direc}/{fam}/{query}'
        embed = embed_query(seq_file, tokenizer, model, device)
        try:
            embed = quant2D(embed, 5, 44)  # nxn 1D array
        except ValueError:
            continue

        # Search idct embeddings and analyze results
        results = query_search(embed, 'Data/avg_dct', 100, 'cityblock')
        m, t, c = search_results(f'{fam}/{query}', results)
        (match, top, clan, total) = (match + m, top + t, clan + c, total + 1)
        logging.info('Queries: %s, Matches: %s, Top10: %s, Clan: %s\n', total, match, top, clan)


if __name__ == '__main__':
    main()
