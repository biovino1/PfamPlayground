"""================================================================================================
This script averages all embeddings for each Pfam family and saves the average embedding as a numpy
array in a new directory.

Ben Iovino  05/12/23   PfamPlayground
================================================================================================"""

import os
import numpy as np
from Bio import SeqIO


def get_seqs(family: str) -> dict:
    """=============================================================================================
    This function accepts a family name and returns a dictionary of sequences in the family.

    :param family: name of Pfam family
    :return dict: seq id is key with sequence as value
    ============================================================================================="""

    sequences = {}
    for file in os.listdir(f'families/{family}'):  # Open each fasta file
        with open(f'families/{family}/{file}', 'r', encoding='utf8') as file:
            for record in SeqIO.parse(file, 'fasta'):
                sequences[record.id] = record.seq  # Add sequence to dictionary

    return sequences


def cons_pos(sequences: dict) -> dict:
    """=============================================================================================
    This function accepts a a dictionary of sequences, one of the sequences being a consensus. It
    returns a dict with the positions that are included in the consensus for each sequence.

    :param sequences: dictionary of sequences
    :return dict: seq id is key with list of positions as value
    ============================================================================================="""

    # Get consensus sequence and remove it so it is not compared to itself
    cons_seq = sequences['consensus']
    del sequences['consensus']

    # For each sequence, compare to consensus to get positions of non-gap characters
    positions = {}
    for seqid, seq in sequences.items():  # pylint: disable=W0612
        pos_list = []
        for i, pos in enumerate(cons_seq):
            if pos != '.' and seq[i] != '.':  # If both are not gaps
                pos_list.append(i)
        positions[seqid] = pos_list

    return positions


def get_embed(family: str, sequences: dict) -> dict:
    """=============================================================================================
    This function accepts a family name, a dictionary of sequences in that family, and a dictionary
    of positions that are included in the consensus sequence. It returns a dictionary of embeddings
    corresponding to the consensus positions.

    :param family: name of Pfam family
    :param sequences: dict where seq id is key with sequence as value
    :param positions: dict where seq id is key with list of positions as value
    :return dict: seq id is key with list of embeddings as value
    ============================================================================================="""

    # Load embeddings from file
    embeddings = {}
    for file in os.listdir(f'prott5_embed/{family}'):
        with open(f'prott5_embed/{family}/{file}', 'r', encoding='utf8') as file:
            seqname = file.name.split('/')[-1].split('.')[0]  # To match with sequences dict keys
            embeddings[seqname] = np.loadtxt(file)

    # Pad embeddings to match length of consensus sequence
    del embeddings['consensus']
    for seqid, embed in embeddings.items():
        sequence = sequences[seqid]

        # Pad embeddings with 0s where there are gaps in the aligned sequences
        count, pad_emb = 0, []
        for pos in sequence:
            if pos == '.':
                pad_emb.append(0)
            elif pos != '.':
                pad_emb.append(embed[count])
                count += 1
        embeddings[seqid] = pad_emb

    return embeddings


def average_embed(family: str, positions: dict, embeddings: dict) -> list:
    """=============================================================================================
    This function accepts a dictionary of positions that are included in the consensus sequence and
    a dictionary of embeddings. It returns a list of vectors that represents the average embedding
    for each position in the consensus sequence.

    :param family: name of Pfam family
    :param sequences: dict where seq id is key with sequence as value
    :param positions: dict where seq id is key with list of positions as value
    :return dict: seq id is key with list of embeddings as value
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

    # Move through each position (list of embeddings) and average them
    avg_embed = []
    for pos, embed in seq_embed.items():
        avg_embed.append(np.mean(embed, axis=0))  # Find mean for each position (float)
    np.savetxt(f'prott5_embed/{family}/avg_embed.txt', avg_embed, '%.6e')


def main():
    """=============================================================================================
    Main goes through each Pfam family and calls get_seqs() to get protein sequences, cons_pos() to
    get the consensus sequence positions, get_embed() to get the embeddings for each sequence, and
    average_embed() to average the embeddings and save them to file.
    ============================================================================================="""

    for family in os.listdir('prott5_embed'):

        # Get sequences and their consensus positions
        sequences = get_seqs(family)
        positions = cons_pos(sequences)

        # Get embeddings for each sequence in family and average them
        embeddings = get_embed(family, sequences)
        average_embed(family, positions, embeddings)


if __name__ == '__main__':
    main()
