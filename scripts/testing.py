"""This script is used for testing various aspects of searching and embedding.

__author__ = "Ben Iovino"
__date__ = "07/20/23"
"""

import logging
import os
import numpy as np
import torch
from util import load_model, Embedding, Transform
from Bio import SeqIO
from random import sample
import datetime

log_filename = 'data/logs/testing.log'  #pylint: disable=C0103
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
logging.basicConfig(filename=log_filename, filemode='w',
                     level=logging.INFO, format='%(message)s')


def test_layers():
    """Test each layer of ESM2-t36-3B by embedding sequences from families that were missed in prior
    searches and transforming each family's average embedding with DCT dimensions of 5x44. Sequences
    from the full pfam db are then searched against these dct representations and the results are
    logged.
    """

    # Embed seqs, transform them, and search
    for i in range(1, 36):
        os.system(f'python scripts/embed_pfam.py -d data/ -e esm2 -l {i}')
        os.system(f'python scripts/dct_avg.py -d data/esm2_{i}_embed -s1 5 -s2 44')
        os.system(f'python scripts/search_dct.py -d data/esm2_{i}_544_avg.npy -l {i} -s1 5 -s2 44')

        # Read last line of search log and write to test_layers log
        with open('data/logs/search_dct.log', 'r', encoding='utf8') as file:
            lines = file.readlines()
            last_line = lines[-2]
        logging.info('Search results for ESM2 layer %s\n%s\n', i, last_line)
        os.remove(f'esm2_{i}_embed')
        os.remove(f'esm2_{i}_544_avg.npy')


def test_transforms():
    """Test different dct transforms by embedding sequences with best layers from test_layers() and
    transforming each family's average embedding with different DCT dimensions. Sequences from the
    full pfam db are then searched against these dct representations and the results are logged.
    """

    # DCT transforms to test
    i = [3, 4, 5, 6, 7, 8]
    j = [20, 30, 40, 50, 60, 70, 80]

    # Embed using best layers from test_layers()
    for lay in [17, 25]:

        # For every combination of i and j, transform embeddings
        for s1 in i:
            for s2 in j:

                # Calculate average dct embedding for each family
                os.system('python scripts/dct_avg.py '
                    f'-d data/esm2_{lay}_embed -s1 {s1} -s2 {s2}')

                # Search against the full pfam db
                os.system('python scripts/search_dct.py '
                    f'-d data/esm2_{lay}_{s1}{s2}_avg.npy -l {lay} -s1 {s1} -s2 {s2}')

                # Read last line of search log and write to test_transforms log
                with open('data/logs/search_dct.log', 'r', encoding='utf8') as file:
                    lines = file.readlines()
                    last_line = lines[-2]
                logging.info('Search results for ESM2 layer %s with DCT dimensions %sx%s\n%s\n',
                    lay, s1, s2, last_line)
                os.system(f'rm data/esm2_{lay}_{s1}{s2}_avg.npy')


def embed_query(
    sequence: str, tokenizer, model, device: str, encoder: str, layer: int) -> np.ndarray:
    """Returns the embedding of a fasta sequence.

    :param sequence: path to fasta file containing query sequence
    :param tokenizer: tokenizer
    :param model: encoder model
    :param device: cpu or gpu
    :param encoder: prott5 or esm2
    :param layer: layer to extract features from (if using esm2)
    :return: embedding of query sequence
    """

    # Get seq from file
    seq = ()
    with open(sequence, 'r', encoding='utf8') as f:
        for seq in SeqIO.parse(f, 'fasta'):
            seq = (seq.id, str(seq.seq))

    # Initialize Embedding object and embed sequence
    embed = Embedding(seq[0], seq[1], None)
    embed.embed_seq(tokenizer, model, device, encoder, layer)

    return embed.embed[1]


def get_transform(seq: str, tokenizer, model, device: str, transforms, layer) -> Transform:
    """Returns the DCT of an embedded fasta sequence.

    :param seq: path to fasta file containing query sequence
    :param tokenizer: tokenizer
    :param model: encoder model
    :param device: cpu or gpu
    :return: Transform object containing dct representation of query sequence
    """

    # Get DCT for each layer
    query = '/'.join(seq.split('/')[2:])
    embed = embed_query(seq, tokenizer, model, device, 'esm2', layer)
    embed = Transform(query, np.array(embed), None)
    embed.quant_2D(transforms[0], transforms[1])
    if embed.trans[1] is None:  # Skip if DCT is None
        return None  #\\NOSONAR

    return embed


def test_search(): #\\NOSONAR
    """Testing search method in Transform class.
    """

    # Load tokenizer and encoder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # pylint: disable=E1101
    tokenizer, model = load_model('esm2', device)

    db1 = np.load('data/esm2_13_880_avg.npy', allow_pickle=True)
    db2 = np.load('data/esm2_17_875_avg.npy', allow_pickle=True)
    db3 = np.load('data/esm2_25_880_avg.npy', allow_pickle=True)
    dbs = [db1, db2, db3]
    transforms = [(8, 80), (8, 75), (8, 80)]
    layers = [13, 17, 25]
    thresholds = [6.08, 6.80, 7.48]

    direc = 'data/full_seqs'
    for fam in os.listdir(direc):
        results_list = []
        for i, database in enumerate(dbs):
            search_fams = [transform[0] for transform in database]
            if fam not in search_fams:
                break

            # Randomly sample one query from family and get it appropriate dct
            queries = os.listdir(f'{direc}/{fam}')
            query = sample(queries, 1)[0]
            seq_file = f'{direc}/{fam}/{query}'
            query = get_transform(seq_file, tokenizer, model, device, transforms[i], layers[i])
            if query is None:
                logging.info('%s\n%s\nQuery was too small for transformation dimensions',
                          datetime.datetime.now(), query)
                break

            # Search for query in database
            results = query.search(database, 100)
            if results[0][1] > thresholds[i]:
                print(f'Query {query.trans[0]} was found in {fam} with score {results[0][1]}')
                break
            else:
                top_res = results[0:5]
                top_res = [res[0] for res in top_res]
                results_list.extend(top_res)
                if i == 2:
                    print(f'Query {query.trans[0]} not found, results: {results_list}')


def main():
    """Main calls test functions.
    """


    test_search()


if __name__ == '__main__':
    main()
