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


def get_cos_sim(family: str, embeddings: dict) -> tuple[dict, list]:
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
    avg_cos = []
    for pos, embed in embeddings.items():
        cos_sim[pos] = []
        avg_vec = avg_embed[pos]  # Average embedding for that position

        # Calculate cosine similarity between average embedding and each embedding for that position
        for emb in embed:
            sim = np.dot(avg_vec, emb) / (np.linalg.norm(avg_vec) * np.linalg.norm(emb))
            cos_sim[pos].append(sim)

        # Get average cosine similarity
        avg_cos.append(np.mean(cos_sim[pos]))

    return cos_sim, avg_cos


def plot_regions(family: str, cos_sim: dict, avg_cos: list):
    """=============================================================================================
    This function accepts a dictionary of cosine similarities and a list of their average values and
    plots the variance and cosine similarity for each position.

    :param family: name of Pfam family
    :param cos_sim: dict where position is key with list of embeddings from each seq as value
    :param avg_cos: list of average cosine similarities for each position
    ============================================================================================="""

    # Find inverse of variance for each position
    var = []
    for sim in cos_sim.values():
        var.append((-1)*(np.var(sim)))

    # Multi figure plot, one for variance and one for cosine similarity
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    fig.suptitle(f'Variance and Cosine Similarity for {family}')

    # Plot variance
    axs[0].plot(var)
    axs[0].set_title('Variance')
    axs[0].set_xlabel('Position')
    axs[0].set_ylabel('Variance')

    # Plot cosine similarity
    axs[1].plot(avg_cos)
    axs[1].set_title('Cosine Similarity')
    axs[1].set_xlabel('Position')
    axs[1].set_ylabel('Cosine Similarity')

    # Display figure
    #plt.show()


def determine_regions(cos_sim: dict, avg_cos: list) -> dict:
    """=============================================================================================
    This function accepts a dictionary of cosine similarities and returns a dictionary of regions,
    each one being consecutive positions with relatively high cosine similarity. A region is defined
    as having at least 2 positions with cosine similarity greater than the mean plus one standard
    deviation.

    :param cos_sim: dict where position is key with list of embeddings from each seq as value
    :param avg_cos: list of average cosine similarities for each position
    :return dict: region is key with list of positions and average cosine similarity as value
    ============================================================================================="""

    # Base high cosine similarity off of mean cosine similarity
    #print(f'Mean cos_sim: {np.mean(avg_cos)}')
    #print(f'Std cos_sim: {np.var(avg_cos)}')
    #print(len(cos_sim.keys()))

    # Find continuous regions (>= 2 positions) of high cosine similarity (> mean+1 std)
    regions = {}
    num_regions = 0
    curr_region = []
    in_region = False
    len_region = 0
    sim_region = 0
    for i, sim in enumerate(avg_cos):

        # If cosine similarity is high, add to current region
        if sim >= np.mean(avg_cos) + np.std(avg_cos):
            curr_region.append(i)
            in_region = True
            len_region += 1
            sim_region += sim

        # If cosine similarity is low, end current region
        else:
            if in_region and len_region >= 2:
                num_regions += 1
                regions[num_regions] = [curr_region, sim_region/len_region]
            curr_region = []
            in_region = False
            len_region, sim_region = 0, 0

    return regions


def filter_regions(regions: dict) -> dict:
    """=============================================================================================
    This function accepts a dictionary of regions and returns a filtered dictionary of regions.

    :param regions: dict where region is key with list of positions and average cosine similarity
    :return dict: region is key with list of positions and average cosine similarity as value
    ============================================================================================="""

    # Sort by average cosine similarity, highest to lowest
    regions = dict(sorted(regions.items(), key=lambda item: item[1][1], reverse=True))

    # Select top 3 regions
    regions = dict(list(regions.items())[:3])
    print(regions)


def main():

    for family in os.listdir('prott5_embed'):

        # Get sequences and their consensus positions
        sequences = get_seqs(family)
        positions = cons_pos(sequences)

        # Get embeddings for each sequence in family and take only consensus positions
        embeddings = get_embed(family, sequences)
        cons_embed = embed_pos(positions, embeddings)

        # Find regions of high cosine similarity to consensus embedding
        cos_sim, avg_cos = get_cos_sim(family, cons_embed)
        regions = determine_regions(cos_sim, avg_cos)
        filter_regions(regions)
        plot_regions(family, cos_sim, avg_cos)


if __name__ == '__main__':
    main()
