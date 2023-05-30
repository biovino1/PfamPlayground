"""================================================================================================
This script takes embeddings from a directory and determines positions that are highly similar to
the consensus embedding.

Ben Iovino  05/19/23   PfamPlayground
================================================================================================"""

import os
import matplotlib.pyplot as plt
import numpy as np
from avg_embed import get_seqs, cons_pos, get_embed
from math import ceil


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


def get_cos_sim(family: str, embeddings: dict) -> list:
    """=============================================================================================
    This function accepts a dictionary of embeddings corresponding to positions in the consensus
    sequence and returns a dictionary of cosine similarities between the average embedding and each
    embedding for that position.

    :param family: name of Pfam family
    :param embeddings: dict where position is key with list of embeddings from each seq as value
    :return list: list of average cosine similarities for each position
    ============================================================================================="""

    # Get average embedding
    avg_embed = np.loadtxt(f'Data/avg_embed/{family}/avg_embed.txt')

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

    return avg_cos


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
    plt.show()


def filter_regions(regions: dict) -> dict:
    """=============================================================================================
    This function accepts a dictionary of regions and returns a filtered dictionary of regions.

    :param regions: dict where region is key with list of positions and average cosine similarity
    :return dict: region is key with list of positions and average cosine similarity as value
    ============================================================================================="""

    # Remove regions that are close to beginning or end of sequence
    del_reg = []
    #for reg, pos in regions.items():
        #if pos[0][0] < 3 or pos[0][-1] > (pos[1] - 3):
            #del_reg.append(reg)

    # Sort by average cosine similarity, highest to lowest, so we remove lower ones first
    regions = dict(sorted(regions.items(), key=lambda item: item[1][2], reverse=True))
    mids = {}
    for reg, pos in regions.items():
        mids[reg] = pos[0][ceil((len(pos[0]) - 1)/2)]  # Find middle position for each region

    # Go through middle positions and remove ones that are < 3 positions away
    for reg1, pos1 in mids.items():
        for reg2, pos2 in mids.items():
            if reg1 != reg2 and abs(pos1 - pos2) < 4 and reg2 not in del_reg and reg1 not in del_reg:
                del_reg.append(reg2)  # Second region is less similar than first, remove it

    print(mids)
    if del_reg:
        print(del_reg)
    for reg in del_reg:
        print(regions[reg])

    # Can't delete while iterating through dictionary
    del_reg = list(set(del_reg))
    for reg in del_reg:
        del regions[reg]

    return regions


def determine_regions(avg_cos: list) -> dict:
    """=============================================================================================
    This function accepts list of cosine similarities and returns a dictionary of regions, each one
    being consecutive positions with relatively high cosine similarity. A region is defined as
    having at least 2 positions with cosine similarity greater than the mean plus one standard
    deviation.

    :param avg_cos: list of average cosine similarities for each position
    :return dict: region is key with list of positions and average cosine similarity as value
    ============================================================================================="""

    # Find continuous regions (>= 2 positions) of relatively high cosine similarity
    regions = {}
    num_regions = 0

    # Want at least 3 regions identified per family, may have to lower threshold
    mean = np.mean(avg_cos)
    std = np.std(avg_cos)
    count = 0
    while num_regions < 3 and count < 10:
        curr_region = []
        in_region = False
        len_region, sim_region = 0, 0
        for i, sim in enumerate(avg_cos):

            # If cosine similarity is high, add to current region
            if sim >= (mean + std):
                curr_region.append(i)
                in_region = True
                len_region += 1
                sim_region += sim

            # If cosine similarity is low, end current region
            else:
                if in_region and len_region >= 2:
                    num_regions += 1
                    regions[num_regions] = [curr_region, len(avg_cos), sim_region/len_region]
                curr_region = []
                in_region = False
                len_region, sim_region = 0, 0
                num_regions = len(regions)

            # Filter regions
            regions = filter_regions(regions)

        # Lower threshold and find more regions if not enough found
        count += 1
        std *= 0.50

    print(regions)
    print()
    return regions


def get_anchors(family: str, regions: dict):
    """=============================================================================================
    This function accepts a family to load the average embedding and a dict of regions and writes to
    a file the anchor embeddings for each region.

    :param family: name of Pfam family
    :param regions: dict where region is key with list of positions and average cosine similarity
    ============================================================================================="""

    # Get average embedding
    avg_embed = np.loadtxt(f'Data/avg_embed/{family}/avg_embed.txt')

    # For each set of regions, find anchor residues
    anchors_pos = []
    for reg in regions.values():

        # Get middle position
        mid = ceil((len(reg[0]) - 1)/2)
        anchors_pos.append(reg[0][mid])

    # Grab embeddings from average embedding
    anchor_embed = []
    for pos in anchors_pos:
        anchor_embed.append(avg_embed[pos])

    # Save anchor embeddings to file
    if not os.path.exists(f'Data/anchors/{family}'):
        os.makedirs(f'Data/anchors/{family}')
    np.savetxt(f'Data/anchors/{family}/anchor_embed.txt', anchor_embed, '%.6e')


def main():
    """=============================================================================================
    Main goes through each family that has embeddings and finds anchor residues for each family
    based on cosine similarity to the average embedding.
    ============================================================================================="""

    for family in os.listdir('Data/prott5_embed'):

        if family != 'Keratin':
            continue

        # Get sequences and their consensus positions
        sequences = get_seqs(family)
        positions = cons_pos(sequences)

        # Get embeddings for each sequence in family and take only consensus positions
        embeddings = get_embed(family, sequences)
        cons_embed = embed_pos(positions, embeddings)

        # Find regions of high cosine similarity to consensus embedding
        avg_cos = get_cos_sim(family, cons_embed)  #pylint: disable=W0612
        regions = determine_regions(avg_cos)

        # Sort by average cosine similarity, highest to lowest and take top 3 regions
        regions = dict(sorted(regions.items(), key=lambda item: item[1][2], reverse=True))
        regions = dict(list(regions.items())[:5])

        # Get anchor residues (embedding) for each sequence
        get_anchors(family, regions)


if __name__ == '__main__':
    main()
