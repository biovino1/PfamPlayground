"""Finds the most representative sequence for each Pfam family from the seed database.
Sequence is chosen by finding the most common character at each position in the alignment
and then choosing the sequence with the most common characters.

__author__ = "Ben Iovino"
__date__ = "08/25/23"
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


def most_com(seqs: dict, chars: dict) -> str:
    """Returns most representative sequence for a family.

    :param seqs: dictionary where key is sequence name and value is sequence string
    :param chars: dictionary where key is position and value is dictionary of characters and their
    counts
    :return: most representative sequence
    """

    scores = {}
    for seq in seqs.keys():
        scores[seq] = 0
        sequence = seqs[seq]
        for i, char in enumerate(sequence):
            if char in chars[i]:
                scores[seq] += 1

    return max(scores, key=scores.get)


def find_rep(direc: str):
    """Returns most representative sequence for each family in directory.

    :param direc: directory of families
    """

    # For each family in directory, get most representative sequence
    for fam in os.listdir(direc):

        # Read sequence file
        seqs = {}
        with open(f'{direc}/{fam}/seqs.fa', 'r', encoding='utf8') as f:
            for seq in SeqIO.parse(f, 'fasta'):
                if seq.id == 'consensus':
                    continue
                seqs[seq.id] = str(seq.seq)

        # Count characters at each position and find which sequence has most common characters
        chars = count_chars(seqs)
        rep_seq = most_com(seqs, chars)

        # Write most representative sequence to file
        sequence = seqs[rep_seq].replace('.', '')  # Remove gaps
        os.mkdir(f'data/rep_seqs/{fam}')
        with open(f'data/rep_seqs/{fam}/seq.fa', 'w', encoding='utf8') as f:
            f.write(f'>{rep_seq}\n{sequence}')


def main():
    """Main
    """

    os.mkdir('data/rep_seqs')
    direc = 'data/families_gaps'
    find_rep(direc)


if __name__ == '__main__':
    main()
