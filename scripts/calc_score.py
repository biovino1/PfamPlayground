"""Calculates score for results from a search of a query against a database.

Currently reads in a file of results, but the idea is to have this calculate
the score directly after a search with results in memory.

__author__ = "Ben Iovino"
__date__ = 08/11/2023
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


def load_results(file: str) -> dict:
    """Returns results from search file.

    :param file: path to search file
    :return: dictionary of results
    """

    searches, in_result, key = {}, False, ''
    with open(file, 'r', encoding='utf8') as sfile:
        for line in sfile:
            if 'seqs.fa' in line:  # Query
                key = line.split('/')[0]
                searches[key] = []
                in_result = True
                continue
            if in_result:
                if line.startswith('Queries'):  # End of results
                    in_result = False
                    continue
                result = line.strip('\n').split(',')
                searches[key].append([result[0], int(result[1].strip('[]'))])

    return searches


def dist_stats(scores: list) -> (float, float):
    """Returns mean and std of all values in a list

    :param results: list of scores from a search
    :return: mean and std of all scores
    """

    mean = sum(scores)/len(scores)
    std = np.std(scores)

    return mean, std


def std_dist(scores: list, mean: float, std: float) -> list:
    """Returns a list of standardized scores

    :param scores: list of scores from a search
    :param mean: mean of scores
    :param std: std of scores
    :return: list of standardized scores
    """

    return [round((score-mean)/std, 3) for score in scores]


def main():
    """Main
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='data/logs/search_dct.log')
    args = parser.parse_args()
    results = load_results(args.d)

    # For each result, get values and standardize them
    query_scores = {}
    for query, result in results.items():
        raw_scores = [r[1] for r in result]
        mean, std = dist_stats(raw_scores)
        std_scores = std_dist(raw_scores, mean, std)
        query_scores[query] = [result[0][0], std_scores[0]]

    # Calculate auroc
    y_true = []
    y_score = []
    for query, result in query_scores.items():
        y_true.append(1 if query in result[0] else 0)
        y_score.append(result[1])
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(12, 5))
    plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {round(auc, 3)})')
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    main()
