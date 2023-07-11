"""================================================================================================
This script parses search results from search_anchors or search_dct. Not written to look pretty.

Ben Iovino  07/11/23   SearchEmb
================================================================================================"""


import datetime
import pickle
import os
import logging
import matplotlib.pyplot as plt
import numpy as np
from dct_embed import quant2D
from utility import prot_t5xl_embed, load_model
from scipy.spatial.distance import cityblock

logging.basicConfig(filename='Data/parse_results.log',
                     level=logging.INFO, format='%(message)s')


def get_results(lines: list) -> dict:
    """=============================================================================================
    This function gets results from search_results.log and return a dictionary of results for each
    query.

    :param lines: list of lines from search_results.csv
    :return results: dictionary of results for each query
    ============================================================================================="""

    # Get results for each query
    results = {}
    time, time2, query, avg_time, count = '', '', '', 0, 0
    for line in lines:

        # Skip line inbetween queries
        if line == '\n':
            time2 = time
            time, query = '', ''
            continue

        if line.startswith('Queries'):
            continue

        # Get time
        if time == '':
            time = line.strip('\n')

            # Subtract time from previous query and add diff to avg_time
            if time2 != '':
                time1 = datetime.datetime.strptime(time2, '%Y-%m-%d %H:%M:%S.%f')
                time2 = datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S.%f')
                avg_time += (time2 - time1).total_seconds()
                count += 1

        # Get query
        elif query == '':
            query = line.strip('\n')
            results[query] = {}

        # Get results
        else:
            fam, sim = line.strip('\n').split(',')
            results[query][fam] = sim

    print(f'Average time per query: {avg_time/count}')

    return results


def count_results(results: dict) -> list:
    """=============================================================================================
    This function counts the number of queries that are in the same family as the top result,
    in the same clan as the top result, and in the same family as any result.

    :param results: dictionary of results for each query
    :return missed_queries: list of queries that were not in the same family as the top n results
    ============================================================================================="""

    missed_queries = []
    missedq_length, missedq_count = 0, 0
    match, top, clan, total = 0, 0, 0, 0
    for query, se_res in results.items():
        total += 1

        # See if query is in top results
        query_fam = query.split('/')[0]
        if query_fam == list(se_res.keys())[0].split('/')[0]:
            match += 1
            continue
        if query_fam in [key.split('/')[0] for key in list(se_res.keys())]:
            top += 1
            continue

        # Read clans dict and see if query is in same clan as top result
        with open('Data/clans.pkl', 'rb') as file:
            clans = pickle.load(file)
        for fams in clans.values():
            if query_fam in fams and list(se_res.keys())[0].split('/')[0] in fams:
                clan += 1
                continue

        # Get average sim between query and top n results
        sim = 0
        for sim_val in se_res.values():
            sim += float(sim_val.strip('[]'))
        sim /= len(se_res)

        # Read length of fasta file
        missed_queries.append((query, sim))
        with open(f'Data/full_seqs/{query}', 'r', encoding='utf8') as file:
            lines = file.readlines()
        length = len(''.join([line.strip('\n') for line in lines[1:]]))
        missedq_length += length
        missedq_count += 1

    print(f'Average length of missed query: {missedq_length/missedq_count}')
    print(f'Total: {total}\nMatch: {match}\nTop: {top}\nClan: {clan}')

    return missed_queries


def plot_mqs(num_seqs: list):
    """=============================================================================================
    This function plots a histogram of the number of sequences in each family of missed queries.

    :param num_seqs: list of number of sequences in each family of missed queries
    ============================================================================================="""


    # Plot histogram of number of seqs in family
    plt.hist(num_seqs, bins=50)
    plt.xlabel('Number of sequences in family')
    plt.ylabel('Number of families')
    plt.title('Number of sequences in family of missed queries')
    plt.show()


def mqs(queries: list):
    """=============================================================================================
    This function accepts a list of queries that were not in the same family as the top n results
    and reads the sequence in that pfam family to find the average length of sequences in the
    family, the number of sequences in the family, and the L1 distance between the query and avg dct
    of the family.

    :param queries: list of queries that were not in the same family as the top n results
    ============================================================================================="""

    # Load models for embedding
    tokenizer, model = load_model('prott5', 'cpu')

    avg_len, num_seqs = 0, []
    for query in queries:
        fam = query[0].split('/')[0]
        avg_sim = query[1]

        # Seqs in query fam from Pfam-A.seed
        fam_seqs = os.listdir(f'Data/families_nogaps/{fam}')
        avg_len += len(fam_seqs)
        num_seqs.append(len(fam_seqs))

        # Get avg dct of family
        avg_dct = np.load(f'Data/avg_dct/{fam}/avg_dct.npy')

        # Embed query seq
        with open(f'Data/full_seqs/{query[0]}', 'r', encoding='utf8') as file:
            lines = file.readlines()
        seq = ''.join([line.strip('\n') for line in lines[1:]])

        # Get dct of query seq
        embed = prot_t5xl_embed(seq, tokenizer, model, 'cpu')
        dct = quant2D(embed, 5, 44)

        # Get L1 distance between query and avg dct
        dist = 1/cityblock(avg_dct, dct)

        # Log dist and avg_sim for this query
        logging.info('Query: %s\tL1 between query and avg: %s\tL1 between top 100 results and avg%s'
                     , query[0], dist, avg_sim)

    #plot_mqs(num_seqs)
    print(f'Average number of seqs in family of missed queries: {avg_len/len(queries)}')


def main():
    """=============================================================================================
    ============================================================================================="""

    # Open search_results.csv
    with open('Data/dct_search.log', 'r', encoding='utf8') as file:
        lines = file.readlines()

    # Get results for each query and count results
    results = get_results(lines)
    missed_queries = count_results(results)

    # Compare missed queries to pfam database
    mqs(missed_queries)


if __name__ == '__main__':
    main()
