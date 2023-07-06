"""================================================================================================
This script takes embeddings from a directory and determines positions that are highly similar to
the consensus embedding.

Ben Iovino  05/19/23   SearchEmb
================================================================================================"""

import argparse
import logging
import os
from math import ceil
import matplotlib.pyplot as plt
import numpy as np
from embed_avg import get_seqs, cons_pos, get_embed

logging.basicConfig(filename='Data/get_anchors.log',
                     level=logging.INFO, format='%(message)s')


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
    avg_embed = np.load(f'Data/avg_embed/{family}/avg_embed.npy')

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


def filter_regions(regions: dict, filt_length: int) -> dict:
    """=============================================================================================
    This function accepts a dictionary of regions and returns a filtered dictionary of regions.

    :param regions: dict where region is key with list of positions and average cosine similarity
    :param filt_length: closest distance between regions
    :return dict: region is key with list of positions and average cosine similarity as value
    ============================================================================================="""

    # Sort by average cosine similarity, highest to lowest
    regions = dict(sorted(regions.items(), key=lambda item: item[1][2], reverse=True))
    regions = dict(enumerate(regions.values()))

    # Delete regions at beginning and end of sequence
    del_list = []
    for i, reg in enumerate(regions.values()):
        if reg[0][0] < 3 or reg[0][-1] > (reg[1] - 3):
            del_list.append(i)
    for index in del_list:
        del regions[index]

    # Get middle positions of each region
    mids = []
    for reg in regions.values():
        mids.append(reg[0][ceil((len(reg[0]) - 1)/2)])

    # Filter out regions that are too close to other regions by comparing middle positions
    pointer = 0
    while pointer != len(mids):

        # Create list that links mids to the region they belong to
        mid_regs = list(range(len(mids)))

        # Rename region keys to be consecutive
        regions = dict(enumerate(regions.values()))

        # Comparing middle positions, starting with first one to the rest of the list
        # If any position is within filt_length of the current middle add to delete list
        del_list = []
        for i, num in enumerate(mids[pointer:]):
            if i == 0:  # Skip first one so we can compare to the rest
                continue
            if abs(mids[pointer] - num) < filt_length:
                del_list.append(i + pointer)  # Using separate list to avoid index errors
        pointer += 1

        # Delete indices from list of middle positions
        # Use list of regions corresponding to mids to delete regions from regions dict
        del_count = 0
        for index in del_list:

            # Delete middle position and region
            del mids[index-del_count]
            del regions[mid_regs[index]]
            del_count += 1

    return regions


def determine_regions(avg_cos: list, num_anchors: int, filt_length: int) -> dict:
    """=============================================================================================
    This function accepts list of cosine similarities and returns a dictionary of regions, each one
    being consecutive positions with relatively high cosine similarity. A region is defined as
    having at least 2 positions with cosine similarity greater than the mean plus one standard
    deviation, or lower if not enough regions are found.

    :param avg_cos: list of average cosine similarities for each position
    :param num_anchors: number of anchor residues to find
    :param filt_length: closest distance between regions
    :return dict: region is key with list of positions and average cosine similarity as value
    ============================================================================================="""

    # Find continuous regions (>= 2 positions) of relatively high cosine similarity
    regions = {}
    num_regions = 0
    reg_length = 2

    # Want at least num_anchors identified per family, may have to lower threshold to get them
    mean = np.mean(avg_cos)
    std = np.std(avg_cos)
    count = 0  # Stop after 10 iterations
    while num_regions < num_anchors and count < 10:

        # Going over sequence again, reset region variables
        curr_region, in_region, sim_region = [], False, 0
        for i, sim in enumerate(avg_cos):  # For each position in the embedding

            # If cosine similarity is high, add to current region
            if sim >= (mean + std):
                curr_region.append(i)
                in_region = True
                sim_region += sim

            # If cosine similarity is low, end current region
            else:
                if in_region and len(curr_region) >= reg_length:
                    num_regions += 1
                    regions[num_regions] = [curr_region, len(avg_cos), sim_region/len(curr_region)]
                curr_region, in_region, sim_region = [], False, 0

        # Filter regions up to the third loop, if still not enough regions then lower
        # length of region to 1 and stop filtering for close ones
        if count < 3:
            regions = filter_regions(regions, filt_length)
        else:
            regions = filter_regions(regions, 1)  # Ensure no repeat regions
            reg_length = 1

        # Lower threshold and find more regions if not enough found
        num_regions = len(regions)
        count += 1
        std *= 0.50
    logging.info('Regions: %s', regions)

    return regions


def get_anchors(family: str, regions: dict):
    """=============================================================================================
    This function accepts a family to load the average embedding and a dict of regions and writes to
    a file the anchor embeddings for each region.

    :param family: name of Pfam family
    :param regions: dict where region is key with list of positions and average cosine similarity
    ============================================================================================="""

    # Get average embedding
    avg_embed = np.load(f'Data/avg_embed/{family}/avg_embed.npy')

    # For each set of regions, find anchor residues
    anchors_pos = []
    for reg in regions.values():

        # Get middle position
        mid = ceil((len(reg[0]) - 1)/2)
        anchors_pos.append(reg[0][mid])

    # Log anchor positions
    logging.info('Anchor positions: %s', anchors_pos)

    # Grab embeddings from average embedding
    anchor_embed = []
    for pos in anchors_pos:
        anchor_embed.append(avg_embed[pos])

    # Save anchor embeddings to file
    if not os.path.exists(f'Data/anchors/{family}'):
        os.makedirs(f'Data/anchors/{family}')
    with open(f'Data/anchors/{family}/anchor_embed.npy', 'wb') as emb_f:
        np.save(emb_f, anchor_embed, allow_pickle=True)


def main():
    """=============================================================================================
    Main goes through each family that has embeddings and finds anchor residues for each family
    based on cosine similarity to the average embedding.
    ============================================================================================="""

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', type=int, help='Number of anchor residues to find', default=3)
    parser.add_argument('-f', type=int, help='Closeness of regions to filter out', default=3)
    args = parser.parse_args()

    direc = 'Data/prott5_embed'
    for family in os.listdir(direc):

        # Check if anchors already exist
        if os.path.exists(f'Data/anchors/{family}/anchor_embed.txt'):
            continue
        logging.info('Getting anchor for %s', family)

        # Get sequences and their consensus positions
        sequences = get_seqs(family)
        positions = cons_pos(sequences)

        # Get embeddings for each sequence in family and take only consensus positions
        embeddings = get_embed(f'{direc}/{family}', sequences)
        cons_embed = embed_pos(positions, embeddings)

        # Find regions of high cosine similarity to consensus embedding
        avg_cos = get_cos_sim(family, cons_embed)  #pylint: disable=W0612
        regions = determine_regions(avg_cos, args.a, args.f)

        # Sort by average cosine similarity, highest to lowest and take top num regions
        regions = dict(sorted(regions.items(), key=lambda item: item[1][2], reverse=True))
        regions = dict(list(regions.items())[:args.a])

        # Get anchor residues (embedding) for each sequence
        get_anchors(family, regions)


if __name__ == '__main__':
    main()
