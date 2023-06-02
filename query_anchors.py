"""================================================================================================
This script takes a query sequence and a group of anchor sequences and outputs the most similar
anchor sequences along with their similarity scores.

Ben Iovino  06/01/23   PfamPlayground
================================================================================================"""

import argparse
import os
import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer
from utility import prot_t5xl_embed


def query_search(query: np.ndarray, anchors: str, results: int) -> str:
    """=============================================================================================
    This function takes a query embedding, a directory of anchor embeddings, and finds the most
    similar anchor embedding based on cosine similarity.

    :param query: embedding of query sequence
    :param anchors: directory of anchor embeddings
    :param results: number of results to return
    :return str: name of anchor family with highest similarity
    ============================================================================================="""

    # Search query against every set of anchors
    sims = {}
    for family in os.listdir(anchors):
        anchors = f'Data/anchors/{family}/anchor_embed.txt'
        ancs_emb = np.loadtxt(anchors)

        # Add family to sims dict
        if family not in sims:
            sims[family] = []

        # Compare every anchor embedding to query embedding
        max_sim = 0
        for anchor in ancs_emb:
            cos_sim = []
            for embedding in query:
                cos_sim.append(np.dot(anchor, embedding) /
                        (np.linalg.norm(anchor) * np.linalg.norm(embedding)))
            max_sim = max(cos_sim)  # Find most similar embedding of query to anchor
            sims[family].append(max_sim)

        # Average similarities across all query embeddings to anchor embeddings
        sims[family] = np.mean(sims[family])

    # Sort sims dict and return top 5 results
    top_sims = {}
    for key in sorted(sims, key=sims.get, reverse=True)[:results]:
        top_sims[key] = sims[key]

    return top_sims


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-q', type=str, help='Query sequence') #default='/home/ben/Code/PfamPlayground/Data/families_nogaps/DUF1818/B1X079_CROS5.fa')
    parser.add_argument('-e', type=str, help='Embedding of query sequence') #default='/home/ben/Code/PfamPlayground/Data/prott5_embed/DUF1818/B1X079_CROS5.txt')
    parser.add_argument('-r', type=int, help='Number of results to return', default=5)
    args = parser.parse_args()

    # Load query embedding or embed query sequence
    if args.e:
        query = np.loadtxt(args.e)  # pylint: disable=W0612
    else:

        # Load tokenizer and encoder
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

        # Read fasta file and embed sequence
        with open(args.q, 'r', encoding='utf8') as fa_file:
            seq = ''.join([line.strip('\n') for line in fa_file.readlines()[1:]])
        query = prot_t5xl_embed(seq, tokenizer, model, 'cpu')

    # Search query against every set of anchors
    results = query_search(query, 'Data/anchors', args.r)
    for fam, sim in results.items():
        print(f'{fam}   {round(sim, 4)}')
    print()


if __name__ == '__main__':
    main()
