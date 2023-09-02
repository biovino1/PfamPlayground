"""Reads results from search.log, finds predicted structure of missed queries and
their top results, and measures similarity between them.

__author__ = "Ben Iovino"
__date__ = "08/24/23"
"""

import logging
import os
import subprocess
import esm
import torch
from Bio import SeqIO

log_filename = 'data/logs/comp_str.log'  #pylint: disable=C0103
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
logging.basicConfig(filename=log_filename, filemode='w',
                     level=logging.INFO, format='%(message)s')


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
    """Returns a dict of sequences that represent the query family and top search results.

    Ex. {'PAD_M/A0A669R6D0_PHACC': {'PAD_M': 'DAEPD', 'P4Ha_N': 'ALAML', 'Peptidase_M23_N': 'NHVN'}} 

    :param missed_queries: dictionary of missed queries and their top search results
    :return: dict where key is query name and value is a dict where key is result name and value
    is it's sequence
    """

    seqs = {}
    for key in missed_queries.keys():

        # Get sequence of representative seq for query family
        query_fam = key.split('/')[0]
        seqs[key] = {}
        with open(f'data/rep_seqs/{query_fam}/seq.fa', 'r', encoding='utf8') as f:
            for record in SeqIO.parse(f, 'fasta'):
                seqs[key][query_fam] = str(record.seq)

        # Get representative sequences for top search results
        for result in missed_queries[key]:
            with open(f'data/rep_seqs/{result}/seq.fa', 'r', encoding='utf8') as f:
                for record in SeqIO.parse(f, 'fasta'):
                    seqs[key][result] = str(record.seq)

    return seqs


def esm_fold(seq: str, sfile: str, model):
    """Writes an esm-fold predicted structure to file.

    :param seq: sequence to predict structure of
    :param sfile: file to write structure to
    :param model: esm-fold model
    """

    # Tried multiprocessing but only loaded on one GPU
    if len(seq) > 800:
        logging.info('Sequence too long')
        return

    # Predict and write
    with torch.no_grad():
        output = model.infer_pdb(seq)
    with open(sfile, 'w', encoding='utf8') as pdb:
        pdb.write(output)


def predict_str(model, query_seqs: dict, missed_queries: dict):
    """Writes esm-fold predicted structures to file.

    :param model: esm-fold model
    :param query_seqs: dict where key is query name and value is query sequence
    :param missed_queries: dict where key is query name and value is a dict where key is result
    name and value is it's sequence
    """

    direc = 'data/structures'
    if not os.path.exists(direc):
        os.mkdir(direc)

    # Each query calls 7 predictions: query, query family, and top 5 results from search
    for query, seq in query_seqs.items():
        query_fam, query_seq = query.split('/')[0], query.split('/')[1]

        # Skip set of sequences if already predicted
        if os.path.exists(f'{direc}/{query_fam}'):
            continue
        os.mkdir(f'{direc}/{query_fam}')

        # Predict structure of query sequence
        logging.info('Predicting %s', f'{query_fam}/{query_seq}')
        filename = f'{direc}/{query_fam}/query-{query_seq}.pdb'
        esm_fold(seq, filename, model)

        # Predict representative sequences for query/result families
        count = 0
        for result, result_seq in missed_queries[query].items():
            logging.info('Predicting %s', result)
            filename = f'{direc}/{query_fam}/result{count}-{result}.pdb'
            esm_fold(result_seq, filename, model)
            count += 1


def compare_str() -> dict:
    """Returns a dict of p-values from FATCAT structural alignment.

    return: dict where key is query and value is a dict where key is result and value is p-value
    """

    pvalues = {}
    for direc in os.listdir('data/structures'):

        # Query is first then results are in order
        files = sorted(os.listdir(f'data/structures/{direc}'))
        query = f"{direc}/{files[0].split('-')[1].split('.')[0]}"  # query fam/seq
        pvalues[query] = {}

        # Compare query to each result (result0 is actually query family representative)
        for file in files[1:]:
            result = subprocess.getoutput(f'FATCAT -p1 data/structures/{direc}/{files[0]} '
                      f'-p2 data/structures/{direc}/{file} -q')
            result_name = file.split('-')[1].split('.')[0]
            pvalues[query][result_name] = result.split('\n')[2].split()[1]

    return pvalues


def main():
    """Initializes esm fold model to predict structures of missed queries and their top search
    results. Then compares predicted structures using FATCAT.
    """

    # Get missed queries and their top search results
    results = parse_log('data/logs/search.log')
    missed_queries = parse_results(results)

    # Get their respective sequences
    query_seqs = get_query_seqs(missed_queries)
    missed_queries = get_result_seqs(missed_queries)

    # Load model and predict/compare structures
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #pylint: disable=E1101
    model = esm.pretrained.esmfold_v1()
    model = model.eval().cuda()
    model.to(device)
    predict_str(model, query_seqs, missed_queries)
    compare_str()


if __name__ == '__main__':
    main()
