"""This script is used for testing various aspects of searching and embedding.

__author__ = "Ben Iovino"
__date__ = "07/20/23"
"""

import logging
import os
import datetime
from random import sample
import numpy as np
import torch
from util import load_model, Embedding, Transform
from Bio import SeqIO
from search import search_results
from scipy.spatial.distance import cityblock

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


### TESTING AVERAGE EMBEDDING ###

def embed_query(
    fam: str, tokenizer, model, device: str, query: str) -> tuple:
    """Returns the embedding of a fasta sequence.

    :param fam: family of query sequence
    :param tokenizer: tokenizer
    :param model: encoder model
    :param device: cpu or gpu
    :param query: query sequence
    :return: Embedding and Transform objects
    """

    # Get query sequence from family fasta file
    with open(f'data/families_nogaps/{fam}/seqs.fa', 'r', encoding='utf8') as f:
        for seq in SeqIO.parse(f, 'fasta'):
            if seq.description == query:
                seq_id = seq.id
                seq = str(seq.seq)
                break

    # Initialize Embedding object and embed sequence
    embed = Embedding(seq_id, seq, None)
    embed.embed_seq(tokenizer, model, device, 'esm2', 17)

    # DCT embedding
    transform = Transform(embed.embed[0], embed.embed[1], None)
    transform.quant_2D(8, 75)
    if transform.trans[1] is None:  # Skip if DCT is None
        return None  #\\NOSONAR

    return embed, transform


def test_search(): #\\NOSONAR
    """Testing search method in Transform class.
    """

    # Load tokenizer and encoder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # pylint: disable=E1101
    tokenizer, model = load_model('esm2', device)

    # DCT database
    dct_db = np.load('data/dct_full.npy', allow_pickle=True)

    # List of queries
    queries = {}
    with open('data/queries.txt', 'r', encoding='utf8') as f:
        for line in f:
            queries[line.split('/')[0]] = line.split('/')[1].strip('\n')

    # Search each query against dct database
    counts = {'match': 0, 'top': 0, 'clan': 0, 'total': 0}
    for fam in list(queries.keys()):

        # Get sequence embedding and dct
        embed, dct = embed_query(fam, tokenizer, model, device, queries[fam])
        if dct is None:
            logging.info('%s\n%s\nQuery was too small for transformation dimensions',
                          datetime.datetime.now(), embed.embed[0])
            continue

        # Search dct db - check if top family is same as query family
        results = dct.search(dct_db, 100)
        counts = search_results(f'{fam}/{dct.trans[0]}', results, counts)
        logging.info('DCT: Queries: %s, Matches: %s, Top%s: %s, Clan: %s\n',
                      counts['total'], counts['match'], len(results), counts['top'], counts['clan'])


def avg_dct():
    """
    """

    # Get list of queries that we don't want to add to database
    queries = []
    with open('data/queries.txt', 'r', encoding='utf8') as f:
        for line in f:
            queries.append(line.split('/')[1].strip('\n'))

    # Read all sequences from pfam.seed without including the query
    seqs = {}
    for fam in os.listdir('data/families_nogaps'):
        seqs[fam] = {}
        with open(f'data/families_nogaps/{fam}/seqs.fa', 'r', encoding='utf8') as f:
            for seq in SeqIO.parse(f, 'fasta'):
                if seq.description in queries:
                    continue
                seqs[fam][seq.description] = str(seq.seq)

    # Read all sequences from pfam.full without including query or seqs from pfam.seed
    count = 0
    for fam in os.listdir('data/full_seqs'):
        count += 1
        logging.info('Reading %s', count)
        with open(f'data/full_seqs/{fam}/seqs.fa', 'r', encoding='utf8') as f:
            for seq in SeqIO.parse(f, 'fasta'):
                if len(seqs[fam]) > 500:
                    break
                if seq.description in queries:
                    continue
                if seqs[fam].get(seq.description) is not None:
                    continue
                seqs[fam][seq.description] = str(seq.seq)

    # Remove family if it has less than 10 sequences
    for fam in list(seqs):
        if len(seqs[fam]) < 10:
            del seqs[fam]

    return seqs


def get_dct(seqs):
    """
    """

    if not os.path.exists('data/dct_full'):
        os.makedirs('data/dct_full')

    # Load tokenizer and encoder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # pylint: disable=E1101
    tokenizer, model = load_model('esm2', device)

    # Embed and transform each sequence
    count = 0
    for fam, seqs in seqs.items():

        # Skip if family already exists in dct_full
        if os.path.exists(f'data/dct_full/{fam}.npy'):
            logging.info('Skipping %s', fam)
            continue

        count += 1
        logging.info('Embedding %s, %s', count, fam)
        embeds = []
        for seq in seqs.values():
            embed = Embedding(None, seq, None)
            embed.embed_seq(tokenizer, model, device, 'esm2', 17)
            embeds.append(embed.embed[1])

        # Transform each embedding to DCT
        dcts = []
        for embed in embeds:
            dct = Transform(None, embed, None)
            dct.quant_2D(8, 75)
            dcts.append(dct.trans[1])

        # Average each position across all DCTs using np.mean
        ## ValueError: setting an array element with a sequence
        try:
            avg = np.mean(dcts, axis=0, dtype=np.int16)
            avg = Transform(fam, None, avg)
        except ValueError:
            logging.info('ValueError %s, %s', count, fam)
            continue
        except TypeError:
            logging.info('TypeError %s, %s', count, fam)
            continue

        # Write dct to file
        np.save(f'data/dct_full/{fam}', avg.trans)


def main():
    """Main calls test functions.
    """

    #seqs = avg_dct()
    #get_dct(seqs)

    # Combine all dct's in dct_full into single array
    #dct_db = []
    #for fam in os.listdir('data/dct_full'):
        #dct = np.load(f'data/dct_full/{fam}', allow_pickle=True)
        #dct_db.append(dct)

    # Write dct_db to file
    #np.save('data/dct_full.npy', dct_db)

    test_search()


if __name__ == '__main__':
    main()
