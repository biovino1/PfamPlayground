"""This script takes embeddings from a directory and determines positions that are highly similar to
the consensus embedding.

__author__ = "Ben Iovino"
__date__ = "05/19/23"
"""

import argparse
import logging
import os
from math import ceil
import numpy as np
from embed_avg import get_seqs, cons_pos, get_embed
from util import Embedding

log_filename = 'data/logs/get_anchors.log'  #pylint: disable=C0103
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
logging.basicConfig(filename=log_filename, filemode='w',
                     level=logging.INFO, format='%(message)s')


def embed_pos(positions: dict, embeddings: dict) -> dict:
    """Returns a dictionary of embeddings corresponding to positions in the consensus sequence
    of a Pfam family.

    :param positions: dict where seq id is key with list of positions as value
    :param embeddings: dict where seq id is key with list of embeddings as value
    :return: dict where position is key with list of embeddings as value
    """

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
    """Returns a dictionary of cosine similarities between the average embedding and each
    individual embedding for that position.

    :param family: name of Pfam family
    :param embeddings: dict where position is key with list of embeddings from each seq as value
    :return: list of average cosine similarities for each position
    """

    # Get average embedding
    avg_embed = np.load(f'data/avg_embed/{family}/avg_embed.npy')

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


def get_region(avg_cos: list, regions: dict, threshold: float, num_regions: int) -> dict:
    """Adds a region to the dict if the average cosine similarity of the entire region is
    greater than the threshold.

    :param avg_cos: list of average cosine similarities for each position
    :param regions: dict where region is key with list of positions and average cosine similarity
    :param threshold: minimum average cosine similarity for a position to be added to region
    :param num_regions: number of regions found so far
    :return: dict where region number is key with list of regions and average cosine similarity
    """

    curr_region, in_region, sim_region = [], False, 0
    for i, sim in enumerate(avg_cos):  # For each position in the embedding

        # If cosine similarity is high, add to current region
        if sim >= (threshold):
            curr_region.append(i)
            in_region = True
            sim_region += sim

        # If cosine similarity is low, end current region
        else:
            if in_region and len(curr_region) >= 2:
                num_regions += 1
                regions[num_regions] = [curr_region, len(avg_cos), sim_region/len(curr_region)]
            curr_region, in_region, sim_region = [], False, 0

    return regions


def determine_regions(avg_cos: list, num_anchors: int) -> dict:
    """Returns a dictionary of regions, each one being consecutive positions with relatively
    high cosine similarity. A region is defined as having at least 2 positions with cosine
    similarity greater than the mean plus one standard deviation, or lower if not enough
    regions are found.

    :param avg_cos: list of average cosine similarities for each position
    :param num_anchors: number of anchor residues to find
    :return: dict where key with list of positions and average cosine similarity as value
    """

    # Find continuous regions (>= 2 positions) of relatively high cosine similarity
    regions, num_regions = {}, 0

    # Want at least num_anchors identified per family, may have to lower threshold to get them
    mean, std, count = np.mean(avg_cos), np.std(avg_cos), 0
    while num_regions < num_anchors and count < 10:

        # Get regions with high cosine similarity
        regions = get_region(avg_cos, regions, mean + std, num_regions)

        # Lower threshold and find more regions if not enough found
        num_regions = len(regions)
        count += 1
        std *= 0.50
    logging.info('Regions: %s', regions)

    return regions


def get_anchors(family: str, regions: dict) -> np.ndarray:
    """Writes the anchor residues (embeddings) for each family to a file.

    :param family: name of Pfam family
    :param regions: dict where region is key with list of positions and average cosine similarity
    :return: array of anchor residues (embeddings)
    """

    # Get average embedding
    avg_embed = np.load(f'data/avg_embed/{family}/avg_embed.npy')

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

    # Return anchor embeddings
    return np.asarray(anchor_embed)


def main():
    """Main goes through each family with an average embedding and finds anchor residues for each
    family based on cosine similarity between each embedding in the family to the average embedding.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', type=int, help='Number of anchor residues to find', default=3)
    parser.add_argument('-d', type=str, help='direc of embeds to avg', default='data/esm2_17_embed')
    args = parser.parse_args()

    anchors = []
    for i, family in enumerate(os.listdir(args.d)):

        # Check if anchors already exist
        if os.path.exists(f'data/anchors/{family}/anchor_embed.txt'):
            logging.info('Anchor residues already exist for %s, %s', family, i)
            continue
        logging.info('Getting anchor for %s, %s', family, i)

        # Get sequences and their consensus positions
        sequences = get_seqs(family)
        positions = cons_pos(sequences)

        # Get embeddings for each sequence in family and take only consensus positions
        embeddings = get_embed(f'{args.d}/{family}', sequences)
        cons_embed = embed_pos(positions, embeddings)

        # Find regions of high cosine similarity to consensus embedding
        avg_cos = get_cos_sim(family, cons_embed)  #pylint: disable=W0612
        regions = determine_regions(avg_cos, args.a)

        # Sort by average cosine similarity, highest to lowest and take top num regions
        regions = dict(sorted(regions.items(), key=lambda item: item[1][2], reverse=True))
        regions = dict(list(regions.items())[:args.a])

        # Get anchor residues (embedding) for each sequence
        anchor_embed = get_anchors(family, regions)
        anchor_embed = Embedding(family, None, anchor_embed)
        anchors.append(anchor_embed.embed)

    # Save anchors as one file
    np.save('data/anchors.npy', anchors, allow_pickle=True)


if __name__ == '__main__':
    main()
