"""================================================================================================
This script embeds all of the sequences from Pfam and saves them as numpy arrays.

Ben Iovino  05/08/23   SearchEmb
================================================================================================"""

import argparse
import logging
import os
import torch
import numpy as np
from Bio import SeqIO
from utility import load_model, embed_seq

logging.basicConfig(filename='data/logs/embed_pfam.log',
                     level=logging.INFO, format='%(asctime)s %(message)s')


def load_seqs(file: str) -> list:
    """=============================================================================================
    This function takes a fasta file and returns a list of sequences and their IDs.

    :param file: fasta file
    :return list: list of sequences
    ============================================================================================="""

    # Read each line and add to list
    seqs = []
    with open(file, 'r', encoding='utf8') as f:
        for seq in SeqIO.parse(f, 'fasta'):
            seqs.append((seq.id, str(seq.seq)))

    return seqs


def embed_fam(path: str, tokenizer, model, device, args: argparse.Namespace):
    """=============================================================================================
    This function accepts a directory that contains a fasta file of protein sequences and embeds
    each sequence using the provided tokenizer and encoder. All of the family's embeddings are
    saved as a single numpy array.

    :param path: directory containing fasta files
    :param tokenizer: tokenizer
    :param model: encoder model
    :param device: gpu/cpu
    :param args: directory to store embeddings and encoder type
    ============================================================================================="""

    # Get last directory in path
    fam = path.rsplit('/', maxsplit=1)[-1]
    direc = f'{args.d}/{args.e}_embed'
    if not os.path.isdir(f'{direc}/{fam}'):
        os.makedirs(f'{direc}/{fam}')

    # Get seqs from fasta file
    seqs = load_seqs(f'{path}/seqs.fa')
    embeds = []
    for seq in seqs:  # Embed each sequence individually

        # Check if embeddings already exist
        if os.path.exists(f'{direc}/{fam}/embed.npy'):
            logging.info('Embedding for %s already exists. Skipping...', fam)
            break

        # Skip consensus sequence
        if seq[0] == 'consensus':
            continue

        # Embed and save sequence id and embed in array
        embed = embed_seq(seq, tokenizer, model, device, args)
        embeds.append(np.array([seq[0], embed], dtype=object))  # [(id, embed)]

    # Save embeds to file
    with open(f'{direc}/{fam}/embed.npy', 'wb') as emb:
        np.save(emb, embeds)

    logging.info('Finished embedding sequences in %s\n', fam)


def main():
    """=============================================================================================
    Main loads the tokenizer and encoder models and calls embed_fam() to embed all sequences in each
    family directory from Pfam.
    ============================================================================================="""

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='data')
    parser.add_argument('-e', type=str, default='esm2')
    parser.add_argument('-l', type=int, default=17)
    args = parser.parse_args()

    # Load tokenizer and encoder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #pylint: disable=E1101
    tokenizer, model = load_model(args.e, device)

    # Get names of all family folders and embed all seqs in each one
    families = [f'data/families_nogaps/{fam}' for fam in os.listdir('data/families_nogaps')]
    for fam in families:
        logging.info('Embedding sequences in %s...', fam)
        embed_fam(fam, tokenizer, model, device, args)


if __name__ == '__main__':
    main()
