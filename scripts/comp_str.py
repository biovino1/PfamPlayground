"""Reads results from search.log, finds predicted structure of missed queries and
their top results, and measures similarity between them.

__author__ = "Ben Iovino"
__date__ = "08/24/23"
"""

import os
import time
from Bio import SeqIO


def parse_log(file: str) -> dict:
    """Returns a dict of queries and their top search results.

    :param file: path to search log file
    :return: dictionary where key is named of query and value is list of top search results
    """

    with open(file, 'r', encoding='utf8') as sefile:
        in_result = False
        results = {}
        for line in sefile:

            # Skip line inbetween queries
            if line.startswith(('2023', 'ANCHORS', 'DCT')):
                in_result = False
                continue
            if not in_result:  # Get query family name
                query = line.strip('\n')
                results[query] = []
            else:  # Get top search results
                results[query].append(line.strip('\n').split(',')[0])
            in_result = True

    return results


def parse_results(results: dict):
    """Returns dict of missed queries and their top search results.

    :param results: dictionary of queries and their top search results
    :return: dictionary of missed queries and their top search results
    """

    missed_queries = {}
    for query, result in results.items():
        if query == '':  # Second entry is empty?
            continue
        if query.split('/')[0] != result[0]:  # Top result not query
            missed_queries[query] = result

    return missed_queries


def get_query_seqs(missed_queries: dict) -> dict:
    """Returns a dict of query sequences.

    Ex. {'PAD_M/A0A669R6D0_PHACC': 'SMMSN'}

    :param missed_queries: dictionary of missed queries and their top search results
    :return: dict where key is query name and value is query sequence
    """

    seqs = {}
    for key in missed_queries.keys():
        fam, seq = key.split('/')[0], key.split('/')[1]
        with open(f'data/full_seqs/{fam}/seqs.fa', 'r', encoding='utf8') as f:
            for record in SeqIO.parse(f, 'fasta'):
                if record.id == seq:
                    seqs[key] = str(record.seq)

    return seqs


def get_result_seqs(missed_queries: dict) -> dict:
    """Returns a dict of sequences that represent the top search results.

    Ex. {'PAD_M/A0A669R6D0_PHACC': {'FN3_7': 'DAEPD', 'P4Ha_N': 'ALAML', 'Peptidase_M23_N': 'NHVN'}} 

    :param missed_queries: dictionary of missed queries and their top search results
    :return: dict where key is query name and value is a dict where key is result name and value
    is it's sequence
    """

    seqs = {}
    for key in missed_queries.keys():
        seqs[key] = {}
        for result in missed_queries[key]:

            # Get representative sequence from file
            with open(f'data/rep_seqs/{result}/seq.fa', 'r', encoding='utf8') as f:
                for record in SeqIO.parse(f, 'fasta'):
                    seqs[key][result] = str(record.seq)

    return seqs


def predict_str(query_seqs: dict, missed_queries: dict):
    """Writes esm-fold predicted structures to file.

    :param query_seqs: dict where key is query name and value is query sequence
    :param missed_queries: dict where key is query name and value is a dict where key is result
    name and value is it's sequence
    """

    # Make directory for query with six sequences -> query and top 5 results
    os.mkdir('data/structures')
    for query, seq in query_seqs.items():
        query_fam, query_seq = query.split('/')[0], query.split('/')[1]
        print(query_fam)
        os.mkdir(f'data/structures/{query_fam}')

        # Predict query seq
        os.system(f'curl -X POST --data {seq} https://api.esmatlas.com/foldSequence/v1/pdb/ '
                  f'> data/structures/{query_fam}/{query_seq}.pdb')
        time.sleep(10)  # server is rate limited

        # Predict each result
        for result, result_seq in missed_queries[query].items():
            os.system(f'curl -X POST --data {result_seq} '
                       'https://api.esmatlas.com/foldSequence/v1/pdb/ '
                        f'> data/structures/{query_fam}/{result}.pdb')
            time.sleep(10)
            break


def main():

    # Get missed queries and their top search results
    results = parse_log('data/logs/search.log')
    missed_queries = parse_results(results)

    # Get their respective sequences
    query_seqs = get_query_seqs(missed_queries)
    missed_queries = get_result_seqs(missed_queries)

    # Predict structure of query and top results
    predict_str(query_seqs, missed_queries)



if __name__ == '__main__':
    main()
