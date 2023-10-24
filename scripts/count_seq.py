"""Counts the number of characters for each sequence at each position in each Pfam family from
the seed database. Can either write the sequence with the most common characters (representative)
or the sequence with the median number of characters (used as a query later on).

__author__ = "Ben Iovino"
__date__ = "10/24/23"
"""

import os
from Bio import SeqIO


def count_chars(seqs: dict) -> dict:
    """Returns a dictionary of characters at each position in each sequence.

    :param seqs: dictionary where key is sequence name and value is sequence string
    return: dictionary where key is position and value is dictionary of characters and their counts
    """

    chars = {}
    for seq in seqs.values():
        for i, char in enumerate(seq):
            if i not in chars:  # Initialize dict for position
                chars[i] = {}
            if char not in chars[i]:
                chars[i][char] = 0
            chars[i][char] += 1

    # Find max occuring character at each position - if tie, keep all
    for pos, counts in chars.items():
        max_count = max(counts.values())
        chars[pos] = [char for char, count in counts.items() if count == max_count]

    return chars


def score_seqs(seqs: dict, chars: dict) -> dict:
    """Returns a dictionary of scores for each sequence where the score is the number of
    characters in the sequence that match the most common character at each position.

    :param seqs: dictionary where key is sequence name and value is sequence string
    :param chars: dictionary where key is position and value is dictionary of characters and their
    counts
    :return dict: dictionary where key is sequence name and value is score
    """

    scores = {}
    for seq in seqs.keys():
        scores[seq] = 0
        sequence = seqs[seq]
        for i, char in enumerate(sequence):
            if char in chars[i]:
                scores[seq] += 1

    return scores


def parse_fasta(fasta: str) -> dict:
    """Returns dictionary of sequences from fasta file.

    :param seqs: dictionary where key is sequence name and value is fasta string
    :param fasta: fasta file path
    :return dict: updated seqs dict
    """

    seqs = {}
    with open(fasta, 'r', encoding='utf8') as ffile:
        for seq in SeqIO.parse(ffile, 'fasta'):
            if seq.id == 'consensus':
                continue
            seqs[seq.description] = str(seq.seq)

    return seqs


def get_rep(scores: dict, seqs: dict, fam: str):
    """Writes the most representative sequence to file.

    :param scores: dictionary where key is sequence name and value is score
    :param seqs: dictionary where key is sequence name and value is sequence string
    :param fam: family name
    """

    direc = 'data/rep_seqs'
    if not os.path.exists(direc):
        os.mkdir(direc)

    # Get most representative sequence and write
    rep_seq = max(scores, key=scores.get)
    sequence = seqs[rep_seq].replace('.', '')  # Remove gaps
    with open(f'{direc}/{fam}.fa', 'w', encoding='utf8') as f:
        f.write(f'>{rep_seq}\n{sequence}')


def find_seq(direc: str, seq: str):
    """Writes desired sequence from each family to file

    :param direc: directory of families
    :param seq: 'rep' or 'med' for representative or median sequence
    """

    # For each family in directory, get most representative sequence
    for fam in os.listdir(direc):

        # Read seqs, count chars, and score sequences
        seqs = parse_fasta(f'{direc}/{fam}/seqs.fa')
        chars = count_chars(seqs)
        scores = score_seqs(seqs, chars)

        # Write sequence of interest to file
        if seq == 'rep':
            get_rep(scores, seqs, fam)
        if seq == 'med':
            med_seq = sorted(scores, key=scores.get)[len(scores) // 2]
            with open('data/queries.txt', 'a', encoding='utf8') as f:
                f.write(f'{fam}/{med_seq}\n')


def main():
    """Main
    """

    seq = 'med'  # 'rep' or 'med'
    direc = 'data/families_gaps'
    find_seq(direc, seq)


if __name__ == '__main__':
    main()
