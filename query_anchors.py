"""================================================================================================
This script takes a query sequence and a group of anchor sequences and outputs the most similar
anchor sequences along with their similarity scores.

Ben Iovino  06/01/23   SearchEmb
================================================================================================"""

import argparse
import numpy as np
import os
import torch
from transformers import T5EncoderModel, T5Tokenizer
from utility import prot_t5xl_embed


def prefilter(anchor: np.ndarray, query: np.ndarray) -> bool:
    """=============================================================================================
    This function takes an anchor and query embedding and finds the similarity of the anchor to a
    subset of embeddings in the query to determine if it is worth comparing the rest of the query

    :param anchor: embedding of anchor sequence
    :param query: list of embeddings of query sequence
    :return bool: True if anchor is similar to one of the query embeddings, False otherwise
    ============================================================================================="""

    n = len(query) // 3
    for embedding in query[::n]:  # Every nth embedding
        sim = np.dot(anchor, embedding) / \
            (np.linalg.norm(anchor) * np.linalg.norm(embedding))
        if sim > 0.025:
            return True

    return False


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

        # I think np.loadtxt loads a single line as a 1D array, so convert to 2D or else
        # max_sim = max(cos_sim) will throw an error
        if len(ancs_emb) == 1024:
            ancs_emb = [ancs_emb]

        # Add family to sims dict
        if family not in sims:
            sims[family] = []

        # Compare every anchor embedding to query embedding
        max_sim = 0
        for anchor in ancs_emb:
            metric = []

            # If similarity is high enough, compare every embedding to anchor
            if prefilter(anchor, query):
                for embedding in query:

                    # Cosine similarity between anchor and query embedding
                    sim = np.dot(anchor, embedding) / \
                        (np.linalg.norm(anchor) * np.linalg.norm(embedding))
                    metric.append(sim)

                    # Euclidean distance between anchor and query embedding
                    #dist = (1/np.linalg.norm(anchor-embedding))
                    #metric.append(dist)
                    #all_sims.append(dist)

                max_sim = max(metric)  # Find most similar embedding of query to anchor
                sims[family].append(max_sim)

                # Compare similarity to first anchor to average similarities of rest of results
                if len(sims[family]) == 1 and len(sims) > 100:
                    if max_sim < np.mean(list(sims.values())[100]):
                        break  # If similarity is too low, stop comparing to rest of anchors

        # Average similarities across all query embeddings to anchor embeddings
        if len(sims[family]) == 0:
            del sims[family]
        else:
            sims[family] = np.mean(sims[family])

    # Sort sims dict and return top n results
    top_sims = {}
    for key in sorted(sims, key=sims.get, reverse=True)[:results]:
        top_sims[key] = sims[key]

    return top_sims


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-q', type=str, help='Query sequence')
    parser.add_argument('-e', type=str, help='Embedding of query sequence')
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

    # Search query against anchors
    results = query_search(query, 'Data/anchors', args.r)  #pylint: disable=W0612


if __name__ == '__main__':
    main()
