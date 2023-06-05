"""================================================================================================
This script takes a query sequence and a list of anchor sequences and searches for the most
similar anchor sequence.

Ben Iovino  05/24/23   PfamPlayground
================================================================================================"""

import numpy as np
import os
import torch
from random import sample
from transformers import T5EncoderModel, T5Tokenizer
from utility import prot_t5xl_embed


def load_models():
    """=============================================================================================
    This function loads the ProtT5-XL model and tokenizer.

    :return tokenizer, model: tokenizer and model
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

    return tokenizer, model


def embed_query(direc, fam, query, tokenizer, model):
    """=============================================================================================
    This embeds a query sequence and saves it to file. query_anchors.py can also perform this
    function when a fasta file is passed through, but this is more efficient when running multiple
    queries so the models only have to load once.

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
    np.savetxt(f'{direc}/{fam}/{query}.txt', embed)  # query needs file as param


def main():

    # Call query_anchors for every query sequence in a folder
    tokenizer, model = load_models()
    direc = 'Data/full_seqs'
    for fam in os.listdir(direc):

        # Randomly sample one query from family
        queries = os.listdir(f'{direc}/{fam}')
        query = sample(queries, 1)[0]
        embed_query(direc, fam, query, tokenizer, model)

        # Call query_anchors with embedding
        os.system(f'python query_anchors.py -e {direc}/{fam}/{query}.txt')
        os.remove(f'{direc}/{fam}/{query}.txt')  # Embeddings take up a lot of space


if __name__ == '__main__':
    main()
