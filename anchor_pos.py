"""================================================================================================
This script takes embeddings from a directory and determines positions that are highly similar to
the consensus embedding.

Ben Iovino  05/19/23   PfamPlayground
================================================================================================"""

import os
import matplotlib.pyplot as plt
import numpy as np
from avg_embed import get_seqs, cons_pos, get_embed


def embed_pos(positions: dict, embeddings: dict) -> list:
    """=============================================================================================
    This function accepts a dictionary of positions that are included in the consensus sequence and
    a dictionary of embeddings. It returns a list of vectors that correspond to the consensus
    positions.

    :param sequences: dict where seq id is key with sequence as value
    :param positions: dict where seq id is key with list of positions as value
    :return dict: position is key with list of embeddings as value
    ============================================================================================="""

    # Create a dict of lists where each list contains the embeddings for a position in the consensus
    seq_embed = {}
    for seqid, position in positions.items():  # Go through positions for each sequence
        for pos in position:
            if pos not in seq_embed:  # Initialize key and value
                seq_embed[pos] = []
            seq_embed[pos].append(embeddings[seqid][pos])  # Add embedding to list for that pos

    # Sort dictionary by key (out of order for some reason)
    seq_embed = dict(sorted(seq_embed.items()))

    # Set key values to be from 0 to n
    cons_embed = {}
    for i, embed in enumerate(seq_embed.values()):
        cons_embed[i] = embed

    return cons_embed


def get_cos_sim(family: str, embeddings: dict) -> dict:
    """=============================================================================================
    This function accepts a dictionary of embeddings corresponding to positions in the consensus
    sequence and returns a dictionary of cosine similarities between the average embedding and each
    embedding for that position.

    :param family: name of Pfam family
    :param embeddings: dict where position is key with list of embeddings from each seq as value
    :return dict: position is key with list of cosine similarities as value
    ============================================================================================="""

    # Get average embedding
    avg_embed = np.loadtxt(f'prott5_embed/{family}/avg_embed.txt')

    # Get cosine similarity between average embedding and each position
    cos_sim = {}
    for pos, embed in embeddings.items():
        cos_sim[pos] = []
        avg_vec = avg_embed[pos]  # Average embedding for that position

        # Calculate cosine similarity between average embedding and each embedding for that position
        for emb in embed:
            sim = np.dot(avg_vec, emb) / (np.linalg.norm(avg_vec) * np.linalg.norm(emb))*10
            cos_sim[pos].append(sim)

    return cos_sim


def get_regions(family: str, cos_sim: dict):
    """=============================================================================================
    This function accepts a dictionary of cosine similarities and returns a list of positions that
    have low variance.

    :param family: name of Pfam family
    :param cos_sim: dict where position is key with list of embeddings from each seq as value
    ============================================================================================="""

    # Find inverse of variance for each position
    var = []
    for sim in cos_sim.values():
        var.append((-1)*(np.var(sim)))

    # Plot variances
    plt.plot(var)
    plt.title(f'{family} Variance')
    plt.xlabel('Position')
    plt.ylabel('1/Variance')
    plt.show()


def main():

    for family in os.listdir('prott5_embed'):

        # Get sequences and their consensus positions
        sequences = get_seqs(family)
        positions = cons_pos(sequences)

        # Get embeddings for each sequence in family and take only consensus positions
        embeddings = get_embed(family, sequences)
        cons_embed = embed_pos(positions, embeddings)

        # Find regions of high cosine similarity to consensus embedding
        cos_sim = get_cos_sim(family, cons_embed)
        get_regions(family, cos_sim)


if __name__ == '__main__':
    main()
