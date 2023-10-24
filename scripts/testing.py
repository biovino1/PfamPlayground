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
from count_seq import count_chars
from avg_embed import cons_pos

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
    fam: str, tokenizer, model, device: str) -> tuple:
    """Returns the embedding of a fasta sequence.

    :param fam: family of query sequence
    :param tokenizer: tokenizer
    :param model: encoder model
    :param device: cpu or gpu
    :param args: command line arguments
    :return: Embedding and Transform objects
    """

    # Get seqs from file and randomly sample one
    seqs = {}
    with open(f'data/full_seqs/{fam}/seqs.fa', 'r', encoding='utf8') as f:
        for i, seq in enumerate(SeqIO.parse(f, 'fasta')):
            seqs[i] = (seq.id, str(seq.seq))
    seq = sample(list(seqs.values()), 1)[0]

    # Initialize Embedding object and embed sequence
    embed = Embedding(seq[0], seq[1], None)
    embed.embed_seq(tokenizer, model, device, 'esm2', 17)

    # DCT embedding
    transform = Transform(embed.embed[0], embed.embed[1], None)
    transform.quant_2D(8, 75)
    if transform.trans[1] is None:  # Skip if DCT is None
        return None  #\\NOSONAR

    return embed, transform


def search(dct: Transform, search_db: np.ndarray) -> dict:
    """Searches transform against a database of transforms:

    :param database: array of transforms
    :param top: number of results to return
    :return: dict where keys are family names and values are similarity scores
    """

    # Search query against every dct embedding
    sims = {}
    for transform in search_db:
        fam, db_dct = transform[0], transform[1]  # family name, dct vector for family
        dist =  1-cityblock(db_dct, dct.trans[1]) # compare query to dct

        # If distance is within range of family's average dct, add to sims
        rang = repr(transform[2][2]).strip('range(').strip(')')  # my fault for using range object
        if int(rang.split(',')[0]) < dist < int(rang.split(',')[1]):
            sims[fam] = dist

    # Return first n results
    sims = dict(sorted(sims.items(), key=lambda item: item[1], reverse=True))

    return sims


def test_search(): #\\NOSONAR
    """Testing search method in Transform class.
    """

    # Load tokenizer and encoder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # pylint: disable=E1101
    tokenizer, model = load_model('esm2', device)

    # DCT database
    dct_db = np.load('data/dct_stats.npy', allow_pickle=True)
    dct_fams = [transform[0] for transform in dct_db]

    counts = {'match': 0, 'top': 0, 'clan': 0, 'total': 0}
    for fam in dct_fams:

        # Get random sequence from family and embed/transform sequence
        embed, dct = embed_query(fam, tokenizer, model, device)
        if dct is None:
            logging.info('%s\n%s\nQuery was too small for transformation dimensions',
                          datetime.datetime.now(), embed.embed[0])
            continue

        # Search dct db - check if top family is same as query family
        results = search(dct, dct_db)
        counts = search_results(f'{fam}/{dct.trans[0]}', results, counts)
        logging.info('DCT: Queries: %s, Matches: %s, Top%s: %s, Clan: %s\n',
                      counts['total'], counts['match'], len(results), counts['top'], counts['clan'])


### TESTING AVERAGE EMBEDDING ###


def most_avg(seqs: dict, chars: dict, fam: str) -> str:
    """Returns most average sequence for a family based on character counts at each position.
    """

    scores = {}
    for seq in seqs.keys():
        scores[seq] = 0
        sequence = seqs[seq]
        for i, char in enumerate(sequence):
            if char in chars[i]:
                scores[seq] += 1

    # Sort scores and find sequence in the middle
    scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
    seq = list(scores.keys())[len(scores)//2]

    # Write scores to file
    with open('data/queries.txt', 'a', encoding='utf8') as qfile:
        qfile.write(f'{fam}/{seq}\n')

    return seq


def get_seqs(family: str, seqs: dict) -> dict:
    """Returns a dictionary of sequences for a given Pfam family.

    :param family: name of Pfam family
    :param seqs: dict where seq id is key with sequence as value
    :return: seq id is key with sequence as value
    """

    sequences = {}
    with open(f'data/families_gaps/{family}/seqs.fa', 'r', encoding='utf8') as file:
        for record in SeqIO.parse(file, 'fasta'):
            if record.id == seqs[family]:
                continue
            sequences[record.id] = record.seq  # Add sequence to dictionary

    return sequences


def get_embed(direc: str, sequences: dict, avg_seqs, fam) -> dict:
    """Returns a dictionary of embeddings corresponding to the consensus positions for
    each sequence.

    :param family: name of Pfam family
    :param sequences: dict where seq id is key with sequence as value
    :return: dict where seq id is key with list of embeddings as value
    """

    # Load embeddings from file
    embeddings = {}
    embed = np.load(f'{direc}/embed.npy', allow_pickle=True)
    for sid, emb in embed:
        if avg_seqs[fam] == sid:
            print(f'skipping {sid} in {fam}')
            continue
        embeddings[sid] = emb

    # Pad embeddings to match length of consensus sequence
    for seqid, embed in embeddings.items():
        sequence = sequences[seqid]

        # Pad embeddings with 0s where there are gaps in the aligned sequences
        count, pad_emb = 0, []
        for pos in sequence:
            if pos == '.':
                pad_emb.append(0)
            elif pos != '.':
                pad_emb.append(embed[count])
                count += 1
        embeddings[seqid] = pad_emb

    return embeddings


def average_embed(family: str, positions: dict, embeddings: dict):
    """Saves a list of vectors that represents the average embedding for each
    position in the consensus sequence for each Pfam family.

    :param family: name of Pfam family
    :param sequences: dict where seq id is key with sequence as value
    :param positions: dict where seq id is key with list of positions as value
    """

    # Create a dict of lists where each list contains the embeddings for a position in the consensus
    seq_embed = {}
    for seqid, position in positions.items():  # Go through positions for each sequence
        for pos in position:
            if pos not in seq_embed:  # Initialize key and value
                seq_embed[pos] = []
            seq_embed[pos].append(embeddings[seqid][pos])  # Add embedding to list for that pos

    # Sort dictionary by key (out of order for some reason)
    seq_embed = dict(sorted(seq_embed.items()))

    # Move through each position (list of embeddings) and average them
    avg_embed = []
    for pos, embed in seq_embed.items():
        avg_embed.append(np.mean(embed, axis=0))  # Find mean for each position (float)

    # Perform idct on avg_embed
    avg_embed = Transform(family, np.array(avg_embed), None)
    try:
        avg_embed.quant_2D(8, 75)
    except ValueError:
        return None

    return avg_embed.trans


def mid_seq(direc: str):
    """Getting most average sequence for each family in directory
    """

    # For each family in directory, get most average sequence
    avg_seqs = {}
    dcts = []
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
        seq = most_avg(seqs, chars, fam)
        avg_seqs[fam] = seq

        # Get sequences and their positions in consensus
        sequences = get_seqs(fam, avg_seqs)
        positions = cons_pos(sequences)

        # Get embeddings
        embed_direc = f'data/esm2_17_embed/{fam}'
        embeddings = get_embed(embed_direc, sequences, avg_seqs, fam)
        transform = average_embed(fam, positions, embeddings)
        if isinstance(transform, np.ndarray):
            dcts.append(transform)

    # Save avg transform to file
    np.save('data/TEST_esm2_17_875_avg.npy', dcts)


def main():
    """Main calls test functions.
    """


    mid_seq('data/families_gaps')


if __name__ == '__main__':
    main()
