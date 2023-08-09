"""This script embeds sequences from Pfam-A.seed and saves them as numpy arrays.

__author__ = "Ben Iovino"
__date__ = "05/08/23"
"""

import argparse
import logging
import os
import torch
import torch.multiprocessing as mp
import numpy as np
from Bio import SeqIO
from util import load_model, Embedding, Transform

log_filename = 'data/logs/embed_pfam.log'  #pylint: disable=C0103
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
logging.basicConfig(filename=log_filename, filemode='w',
                     level=logging.INFO, format='%(asctime)s %(message)s')


def load_seqs(file: str) -> list:
    """Returns a list of sequences and their IDs froma fasta file.

    :param file: fasta file
    :return list: list of sequences
    """

    # Read each line and add to list
    seqs = []
    with open(file, 'r', encoding='utf8') as f:
        for seq in SeqIO.parse(f, 'fasta'):
            seqs.append((seq.id, str(seq.seq)))

    return seqs


def embed_fam(path: str, tokenizer, model, device, args: argparse.Namespace):
    """Embeds a directory of fasta files and saves the embeddings to a single file.

    :param path: directory containing fasta files
    :param tokenizer: tokenizer
    :param model: encoder model
    :param device: gpu/cpu
    :param args: directory to store embeddings and encoder type/layer
    """

    # Get last directory in path
    fam = path.rsplit('/', maxsplit=1)[-1]
    direc = f'{args.d}/{args.e}_{args.l}_{args.t}'
    if not os.path.isdir(f'{direc}/{fam}'):
        os.makedirs(f'{direc}/{fam}')

    # Get seqs from fasta file
    seqs = load_seqs(f'{path}/seqs.fa')
    embeds = []
    for seq in seqs:  # Embed each sequence individually

        # Check if embeddings already exist
        if os.path.exists(f'{direc}/{fam}/{args.t}.npy'):
            logging.info('Embeddings for %s already exists. Skipping...\n', fam)
            return  # break out of function or else empty list will be saved

        # Skip consensus sequence
        if seq[0] == 'consensus':
            continue

        # Initialize Embedding object and embed sequence
        embed = Embedding(seq[0], seq[1], None)
        embed.embed_seq(tokenizer, model, device, args.e, args.l)

        # Transform embeddings if arg is passed
        if args.t == 'transform':
            embed = Transform(seq[0], embed.embed[1], None)  #pylint: disable=E1136
            embed.quant_2D(args.s1, args.s2)
            if embed.trans[1] is not None:  # Embedding may be too short
                embeds.append(embed.trans)
        else:
            embeds.append(embed.embed)

    # Save embeds to file
    with open(f'{direc}/{fam}/{args.t}.npy', 'wb') as emb:
        np.save(emb, embeds)
    logging.info('Finished embedding sequences in %s\n', fam)


def embed_cpu(args: argparse.Namespace):
    """Embeds pfam sequences using cpu (automatically multi-threaded).

    :param args: explained in main()
    """

    # Load tokenizer and encoder
    tokenizer, model = load_model(args.e, 'cpu')

    families = [f'{args.f}/{fam}' for fam in os.listdir(args.f)]
    for fam in families:
        logging.info('Embedding sequences in %s with %s...', fam, args.e)
        embed_fam(fam, tokenizer, model, 'cpu', args)


def queue_fam(rank: int, queue: mp.Queue, args: argparse.Namespace):
    """Goes through queue of families to embed and transform and which GPU to load models on.

    :param rank: GPU to load model on
    :param queue: queue of families to embed and transform
    :param args: explained in main()
    """

    # Load tokenizer and encoder
    device = torch.device(f'cuda:{rank}')  #pylint: disable=E1101
    tokenizer, model = load_model(args.e, device)

    # Embed and transform each family until queue is empty
    while True:
        fam = queue.get()
        if fam is None:
            break
        logging.info('Embedding sequences in %s with %s...', fam, args.e)
        embed_fam(fam, tokenizer, model, device, args)


def embed_gpu(args: argparse.Namespace):
    """Embeds pfam sequences using gpu with optional multiprocessing.

    :param args: explained in main()
    """

    mp_queue = mp.Queue()
    processes = []
    for rank in range(args.p):
        proc = mp.Process(target=queue_fam, args=(args.g[rank], mp_queue, args))
        proc.start()
        processes.append(proc)
    for fam in [f'{args.f}/{fam}' for fam in os.listdir(args.f)]:
        mp_queue.put(fam)
    for _ in range(args.p):
        mp_queue.put(None)
    for proc in processes:
        proc.join()


def main():
    """Main calls either embed_cpu or embed_gpu depending on args and embeds sequences from
    families in args.f. Can choose to embed with either prott5 or esm2 and which layer of
    esm2 to use. Can also choose to transform embeddings with DCT, in which case the
    transformations will be saved instead of the embeddings.

    args:
        -c: cpu or gpu
        -d: directory to store embeddings
        -e: encoder type (prott5 or esm2)
        -f: family directory
        -g: list of GPU IDs to use
        -l: encoder layer (only for esm2)
        -p: number of processes (for GPU)
        -s1: columns for DCT
        -s2: rows for DCT
        -t: whether to transform embeddings (embed or transform)
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=str, default='cpu')
    parser.add_argument('-d', type=str, default='data')
    parser.add_argument('-e', type=str, default='esm2')
    parser.add_argument('-f', type=str, default='data/families_nogaps')
    parser.add_argument('-g', type=int, nargs='+', default=[1])
    parser.add_argument('-l', type=int, default=10)
    parser.add_argument('-p', type=int, default=1)
    parser.add_argument('-s1', type=int, default=8)
    parser.add_argument('-s2', type=int, default=80)
    parser.add_argument('-t', type=str, default='transform')
    args = parser.parse_args()

    if args.c == 'cpu':
        embed_cpu(args)
    elif args.c == 'gpu':
        embed_gpu(args)


if __name__ == '__main__':
    main()
