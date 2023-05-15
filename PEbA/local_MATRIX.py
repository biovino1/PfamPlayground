"""================================================================================================
This script takes two protein sequences of varying length and finds the highest scoring local
alignment between the two using a substitution matrix.

Ben Iovino  01/23/23   VecAligns
================================================================================================"""

import argparse
import numpy as np
import blosum as bl
from utility import parse_fasta, write_msf, parse_matrix, write_fasta


def local_align(seq1, seq2, subs_matrix, gopen, gext):
    """=============================================================================================
    This function accepts two sequences, creates a matrix corresponding to their lengths, and
    calculates the score of the alignments for each index. A second matrix is scored so that the
    best alignment can be tracebacked.

    :param seq1: first sequence
    :param seq2: second sequence
    :param subs_matrix: substitution scoring matrix (i.e. BLOSUM62)
    :param gopen: gap penalty for opening a new gap
    :param gext: gap penalty for extending a gap
    return: scoring and traceback matrices of optimal scores for the SW-alignment of sequences
    ============================================================================================="""

    # Initialize scoring and traceback matrix based on sequence lengths
    row_length = len(seq1)+1
    col_length = len(seq2)+1
    score_m = np.full((row_length, col_length), 0)
    trace_m = np.full((row_length, col_length), 0)

    # Score matrix by moving through each index
    gap = False
    for i, char in enumerate(seq1):
        seq1_char = char  # Character in 1st sequence
        for j, char in enumerate(seq2):
            seq2_char = char  # Character in 2nd sequence

            # Preceding scoring matrix values
            diagonal = score_m[i][j]
            horizontal = score_m[i+1][j]
            vertical = score_m[i][j+1]

            # Score pair of residues based off BLOSUM matrix
            matrix_score = subs_matrix[f'{seq1_char}{seq2_char}']

            # Add to matrix values via scoring method
            diagonal += matrix_score
            if gap is False:  # Apply gap open penalty if there is no gap
                horizontal += gopen
                vertical += gopen
            if gap is True:  # Apply gap extension penalty if there is a gap
                horizontal += gext
                vertical += gext

            # Assign value to traceback matrix and update gap status
            score = max(diagonal, horizontal, vertical)
            if score == diagonal:
                trace_m[i+1][j+1] = 0
                gap = False
            if score == horizontal:
                trace_m[i+1][j+1] = -1
                gap = True
            if score == vertical:
                trace_m[i+1][j+1] = 1
                gap = True

            # Assign max value to scoring matrix
            score_m[i+1][j+1] = max(score, 0)

    return score_m, trace_m


def traceback(score_m, trace_m, seq1, seq2):
    """=============================================================================================
    This function accepts a scoring and a traceback matrix and two sequences and returns the highest
    scoring local alignment between the two sequences

    :param score_m: scoring matrix
    :param trace_m: traceback matrix
    :param seq1: first sequence
    :param seq2: second sequence
    return: seq1 with gaps, seq2 with gaps, alignment score
    ============================================================================================="""

    # Find index of highest score in scoring matrix, start traceback at this matrix
    high_score_ind = np.unravel_index(np.argmax(score_m, axis=None), score_m.shape)
    align_score = score_m[high_score_ind[0]][high_score_ind[1]]

    # Reverse strings and convert to lists so gaps can be inserted
    rev_seq1 = list(seq1[::-1])
    rev_seq2 = list(seq2[::-1])

    # Move through matrix starting at highest scoring cell
    index = [high_score_ind[0], high_score_ind[1]]

    # Gaps are inserted based on count increasing as we move through sequences
    # If we don't start at bottom right, need to adjust position at which gaps inserted
    count_adjust1 = len(seq1) - high_score_ind[0]
    count_adjust2 = len(seq2) - high_score_ind[1]
    count = 0
    while (index[0] and index[1]) != 0:
        val = trace_m[index[0], index[1]]

        if val == 1:  # If cell is equal to 1, insert a gap into the second sequence
            index[0] = index[0] - 1
            rev_seq2.insert(count+count_adjust2, '.')
        if val == -1:  # If cell is equal to -1, insert a gap into the first sequence
            index[1] = index[1] - 1
            rev_seq1.insert(count+count_adjust1, '.')
        if val == 0:  # If cell is equal to 0, there is no gap
            index[0] = index[0] - 1
            index[1] = index[1] - 1
        count += 1

    # Join lists and reverse strings again
    seq1 = ''.join(rev_seq1)
    seq2 = ''.join(rev_seq2)
    seq1 = seq1[::-1]
    seq2 = seq2[::-1]

    # Introduce gaps at beginning of either sequence based off final index positions
    seq1 = "."*index[1]+seq1
    seq2 = "."*index[0]+seq2

    # Introduce gaps at end of either sequence based off length of other sequence
    align1 = seq1+"."*max(0, len(seq2)-len(seq1))
    align2 = seq2+"."*max(0, len(seq1)-len(seq2))
    return align1, align2, align_score


def main():
    """=============================================================================================
    This function initializes two protein sequences and a scoring matrix, calls SW_align() to get
    the scoring and traceback matrix from SW alignment, calls traceback() to get the local
    alignment, and then writes the alignment to a file in the desired format.
    ============================================================================================="""

    parser = argparse.ArgumentParser()
    parser.add_argument('-f1', '--file1', type=str, help='Name of first fasta file')
    parser.add_argument('-f2', '--file2', type=str, help='Name of second fasta file')
    parser.add_argument('-go', '--gopen', type=float, default=-11, help='Penalty for opening a gap')
    parser.add_argument('-ge', '--gext', type=float, default=-1, help='Penalty for extending a gap')
    parser.add_argument('-m', '--matrix', type=str, default='blosum', help='Substitution matrix to use')
    parser.add_argument('-s', '--score', type=int, default=45, help='Log odds score of subsitution matrix')
    parser.add_argument('-o', '--output', type=str, default='msf', help='Output format')
    parser.add_argument('-sf', '--savefile', type=str, default='n', help='File to save alignment')
    args = parser.parse_args()

    # Parse fasta files for sequences and ids
    seq1, id1 = parse_fasta(args.file1)
    seq2, id2 = parse_fasta(args.file2)

    # Intialize scoring matrix
    if args.matrix == 'blosum':
        matrix = bl.BLOSUM(args.score)
    if args.matrix == 'pfasum':
        matrix = parse_matrix('PFASUM60.txt')

    # Align and traceback
    score_m, trace_m = local_align(seq1, seq2, matrix, args.gopen, args.gext)
    align1, align2, align_score = traceback(score_m, trace_m, seq1, seq2)

    # Write alignment score to file
    align_score /= min(len(seq1), len(seq2))  # Normalize by shortest sequence
    with open('alignments/blosum_align_scores.csv', 'a', encoding='utf8') as file:
        file.write(f'{args.savefile},{align_score}\n')

    # Write align based on desired output format
    if args.output == 'msf':
        write_msf(align1, align2, id1, id2, args.matrix+str(args.score),
                args.gopen, args.gext, args.savefile)
    if args.output == 'fa':
        write_fasta(align1, align2, id1, id2, args.savefile)


if __name__ == '__main__':
    main()
