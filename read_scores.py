"""================================================================================================
This script reads in alignment scores from align_pfam.py to determine homology between sequences.

Ben Iovino  05/19/23   PfamPlayground
================================================================================================"""

import csv
import matplotlib.pyplot as plt


def read_scores(file: str) -> list:
    """=============================================================================================
    This function takes a csv file with alignments and their scores and returns a list of lists
    where the first index is the alignment and the second is the score (for each sublist).

    :param file: csv file
    :return list: list of lists
    ============================================================================================="""

    with open(file, 'r', encoding='utf8') as f:
        reader = csv.reader(f)
        scores = list(reader)
    return scores


def find_homology(scores: list, cutoff: float) -> list:
    """=============================================================================================
    This function accepts a list of alignment scores and returns two lists. The first list is the
    ground truth homology and the second is the homology determined by PEbA alignment scores.

    :param scores: list of lists where the first index is the alignment and the second is the score.
    :param cutoff: cutoff score for homology
    :return list: two lists, one for true homology and other for test homology
    ============================================================================================="""

    # Read alignment to see if they are in same family (i.e. homologous)
    true_homologs = []
    test_homologs = []
    for score in scores:

        # Homology from Pfam families (ground truth)
        align = score[0].split('/')
        if align[2] != '-'.join(align[3].split('-')[1:]):
            true_homologs.append('not homologous')
        else:
            true_homologs.append('homologous')

        # Homology from PEbA alignment score (test)
        if float(score[1]) < cutoff:
            test_homologs.append('not homologous')
        else:
            test_homologs.append('homologous')

    return true_homologs, test_homologs


def compare_homologs(true_homologs: list, test_homologs: list) -> float:
    """=============================================================================================
    This function accepts two lists of homology and returns TPR and FPR.

    :param true_homologs: list of true homology
    :param test_homologs: list of test homology
    :return float: TPR and FPR
    ============================================================================================="""

    # Calculate TP, TN, FP, FN
    tp, tn, fp, fn = 0, 0, 0, 0  # pylint: disable=C0103
    for i in range(len(true_homologs)):  # pylint: disable=C0200
        if true_homologs[i] == 'homologous' and test_homologs[i] == 'homologous':
            tp += 1
        elif true_homologs[i] == 'not homologous' and test_homologs[i] == 'not homologous':
            tn += 1
        elif true_homologs[i] == 'not homologous' and test_homologs[i] == 'homologous':
            fp += 1
        elif true_homologs[i] == 'homologous' and test_homologs[i] == 'not homologous':
            fn += 1

    # Calculate TPR and FPR
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)

    return tpr, fpr


def draw_roc(scores: list):
    """=============================================================================================
    This function accepts a list of alignment scores and draws an ROC curve.

    :param scores: list of alignments and their scores
    ============================================================================================="""

    # Find TPR and FPR for a number of cutoff values
    cutoffs = [i/100 for i in range(301)]
    tprs, fprs = [], []
    for cutoff in cutoffs:
        true_homologs, test_homologs = find_homology(scores, cutoff)
        tpr, fpr = compare_homologs(true_homologs, test_homologs)
        tprs.append(tpr)
        fprs.append(fpr)

    # Reverse lists so they are in order
    tprs.reverse()
    fprs.reverse()

    # Detect when FPR goes from 0 to any value higher
    # AUC1 -> fraction of true homologs found until first non homolog (FPR > 0)
    cutoff, co_tpr, co_fpr = 0, 0, 0
    for i, fpr in enumerate(fprs):
        if fpr > 0:
            cutoff, co_tpr, co_fpr = cutoffs[i], tprs[i], fpr
            print(f'cutoff: {cutoff}, tpr: {co_tpr}, fpr: {co_fpr}')
            break

    # Draw ROC curve
    plt.plot(fprs, tprs, color='red')
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig('roc_curve.png')


def main():
    """=============================================================================================
    Main reads in alignment scores with read_scores() and draws an ROC curve with draw_roc().
    ============================================================================================="""

    # Read csv with align scores and determine homology
    scores = read_scores('alignments/peba_align_scores.csv')
    draw_roc(scores)


if __name__ == '__main__':
    main()
