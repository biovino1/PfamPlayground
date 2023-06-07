"""================================================================================================
This script takes a query sequence and a list of anchor sequences and searches for the most
similar anchor sequence.

Ben Iovino  05/24/23   SearchEmb
================================================================================================"""

import datetime
import logging
import os
import pickle
from random import sample
import torch
from transformers import T5EncoderModel, T5Tokenizer
from utility import prot_t5xl_embed
from query_anchors import query_search

logging.basicConfig(filename='Data/search_results.log',
                     level=logging.INFO, format='%(message)s')


def embed_query(sequence, tokenizer, model, device):
    """=============================================================================================
    This function embeds a query sequence and saves it to file. query_anchors.py can also perform
    this function when a fasta file is passed through, but this is more efficient when running
    multiple queries so the models only have to load once.

    :param sequence: path to fasta file containing query sequence
    :param tokenizer: T5 tokenizer
    :param model: ProtT5-XL model
    :param device: cpu or gpu
    ============================================================================================="""

    # Embed query in this script to save time from loading model every time
    with open(sequence, 'r', encoding='utf8') as fa_file:
        seq = ''.join([line.strip('\n') for line in fa_file.readlines()[1:]])
    embed = prot_t5xl_embed(seq, tokenizer, model, device)

    return embed


def search_results(query: str, results: dict):
    """=============================================================================================
    This function compares a query sequence to a dictionary of results.

    :param query: query sequence
    :param results: dictionary of results from searching query against anchors
    ============================================================================================="""

    # Log time and similarity for each result
    logging.info('%s\n%s', datetime.datetime.now(), query)
    for fam, sim in results.items():
        logging.info('%s,%s', fam, sim)

    # See if query is in top results
    query_fam = query.split('/')[0]
    match, top, clan = 0, 0, 0
    if query_fam == list(results.keys())[0]:  # Top result
        match += 1
        return match, top, clan
    if query_fam in results:  # Top n results
        top += 1
        return match, top, clan

    # Read clans dict and see if query is in same clan as top result
    with open('Data/clans.pkl', 'rb') as file:
        clans = pickle.load(file)
    for fams in clans.values():
        if query_fam in fams and list(results.keys())[0] in fams:
            clan += 1
            return match, top, clan

    return match, top, clan


def main():
    """=============================================================================================
    Main function loads tokenizer and model, randomly samples a query sequence from a family, embeds
    the query, searches the query against anchors, and logs the results
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

    # Call query_anchors for every query sequence in a folder
    match, top, clan, total = 0, 0, 0, 0
    direc = 'Data/full_seqs'
    for fam in os.listdir(direc):

        if fam not in os.listdir('Data/anchors'):
            continue

        # Randomly sample one query from family
        queries = os.listdir(f'{direc}/{fam}')
        query = sample(queries, 1)[0]
        seq_file = f'{direc}/{fam}/{query}'
        embed = embed_query(seq_file, tokenizer, model, device)

        # Search anchors and analyze results
        results = query_search(embed, 'Data/anchors', 10)
        m, t, c = search_results(f'{fam}/{query}', results)
        match += m
        top += t
        clan += c
        total += 1
        logging.info('Queries: %s, Matches: %s, Top10: %s, Clan: %s\n', total, match, top, clan)


if __name__ == '__main__':
    main()
