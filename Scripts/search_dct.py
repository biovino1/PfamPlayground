"""================================================================================================
This script takes a query sequence and a database of dct vectors and outputs the most similar dct
vectors along with their similarity scores.

Ben Iovino  07/05/23   SearchEmb
================================================================================================"""

import datetime
import logging
import os
import pickle
from random import sample
import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer
from utility import prot_t5xl_embed
from idct_embed import quant2D
from scipy.spatial.distance import cityblock

logging.basicConfig(filename='Data/idct_search.log',
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


def query_search(query: np.ndarray, anchors: str, results: int, metric: str) -> str:
    """=============================================================================================
    This function takes a query embedding, a directory of anchor embeddings, and finds the most
    similar anchor embedding based on cosine similarity.

    :param query: embedding of query sequence
    :param anchors: directory of anchor embeddings
    :param results: number of results to return
    :return str: name of anchor family with highest similarity
    ============================================================================================="""

    # Search query against every dct embedding
    sims = {}
    for family in os.listdir(anchors):
        for dct in os.listdir(f'{anchors}/{family}'):

            dct_emb = np.load(f'{anchors}/{family}/{dct}')


    '''
        ancs_emb = np.loadtxt(anchors)

        # I think np.loadtxt loads a single line as a 1D array, so convert to 2D or else
        # max_sim = max(cos_sim) will throw an error
        if len(ancs_emb) == 1024:
            ancs_emb = [ancs_emb]

        # Add family to sims dict
        if family not in sims:
            sims[family] = []

        # Compare every anchor embedding to query embedding
        for anchor in ancs_emb:
            sims = query_sim(anchor, query, sims, family, metric)
        sims[family] = np.mean(sims[family])

    # Sort sims dict and return top n results
    top_sims = {}
    for key in sorted(sims, key=sims.get, reverse=True)[:results]:
        top_sims[key] = sims[key]

    return top_sims
    '''


def main():
    """=============================================================================================
    Main
    ============================================================================================="""

    if os.path.exists('Data/t5_tok.pt'):
        tokenizer = torch.load('Data/t5_tok.pt')
    else:
        tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
        torch.save(tokenizer, 'Data/t5_tok.pt')
    if os.path.exists('Data/prot_t5_xl.pt'):
        model = torch.load('Data/prot_t5_xl.pt')
    else:
        model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
        torch.save(model, 'Data/prot_t5_xl.pt')

    # Load model to gpu if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # pylint: disable=E1101
    model.to(device)
    print(device)

    # Call query_anchors for every query sequence in a folder
    match, top, clan, total = 0, 0, 0, 0
    direc = 'Data/full_seqs'
    for fam in os.listdir(direc):

        if fam not in os.listdir('Data/dct_embed'):
            continue

        # Randomly sample one query from family
        queries = os.listdir(f'{direc}/{fam}')
        query = sample(queries, 1)[0]
        seq_file = f'{direc}/{fam}/{query}'
        #embed = embed_query(seq_file, tokenizer, model, device)
        #embed = quant2D(embed, 8, 8)  # 8x8 DCT

        # Search idct embeddings and analyze results
        query_search(seq_file, 'Data/dct_embed', 10, 'cosine')
        #m, t, c = search_results(f'{fam}/{query}', results)
        #(match, top, clan, total) = (match + m, top + t, clan + c, total + 1)
        #logging.info('Queries: %s, Matches: %s, Top10: %s, Clan: %s\n', total, match, top, clan)


if __name__ == '__main__':
    main()
