"""================================================================================================
This script takes a query sequence and a list of anchor sequences and searches for the most
similar anchor sequence.

Ben Iovino  05/24/23   SearchEmb
================================================================================================"""

import os
import pickle
import torch
from random import sample
from transformers import T5EncoderModel, T5Tokenizer
from utility import prot_t5xl_embed
from query_anchors import query_search


def embed_query(direc, fam, query, tokenizer, model):
    """=============================================================================================
    This function embeds a query sequence and saves it to file. query_anchors.py can also perform
    this function when a fasta file is passed through, but this is more efficient when running
    multiple queries so the models only have to load once.

    :param direc: directory of fasta sequences
    :param fam: family of query sequence
    :param query: query sequence
    :param tokenizer: T5 tokenizer
    :param model: ProtT5-XL model
    ============================================================================================="""

    # Embed query in this script to save time from loading model every time
    with open(f'{direc}/{fam}/{query}', 'r', encoding='utf8') as fa_file:
        seq = ''.join([line.strip('\n') for line in fa_file.readlines()[1:]])
    query = query.replace('.fa', '')
    embed = prot_t5xl_embed(seq, tokenizer, model, 'cpu')

    return embed


def search_results(query: str, results: dict):
    """=============================================================================================
    This function compares a query sequence to a dictionary of results.

    :param query: query sequence
    :param results: dictionary of results from searching query against anchors
    ============================================================================================="""

    # Save results to csv file
    with open('Data/search_results.csv', 'a', encoding='utf8') as csv:
        csv.write(f'{query}\n')
        for fam, sim in results.items():
            csv.write(f'{fam},{sim}\n')
        csv.write('\n')

    # See if query is in top results
    query_fam = query.split('/')[0]
    match, top, clan = 0, 0, 0
    if query_fam == list(results.keys())[0]:
        match += 1
        return match, top, clan
    if query_fam in results:
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
        embed = embed_query(direc, fam, query, tokenizer, model)

        # Search anchors and analyze results
        results = query_search(embed, 'Data/anchors', 10)
        m, t, c = search_results(f'{fam}/{query}', results)
        match += m
        top += t
        clan += c
        total += 1
        print(total, match, top, clan)


if __name__ == '__main__':
    main()
