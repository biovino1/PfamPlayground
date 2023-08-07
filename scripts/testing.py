"""This script is used for testing various aspects of searching and embedding.

__author__ = "Ben Iovino"
__date__ = "07/20/23"
"""

import logging
import os
from random import sample
import numpy as np
import torch
import torch.multiprocessing as mp
from Bio import SeqIO
from util import load_model, Embedding, Transform

NUM_GPUS = torch.cuda.device_count()

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
        os.system('python scripts/dct_avg.py -d data/esm2_embed -s1 5 -s2 44')
        os.system(f'python scripts/search_dct.py -d data/avg_dct.npy -l {i} -s1 5 -s2 44')

        # Read last line of search log and write to test_layers log
        with open('data/logs/search_dct.log', 'r', encoding='utf8') as file:
            lines = file.readlines()
            last_line = lines[-2]
        logging.info('Search results for ESM2 layer %s\n%s\n', i, last_line)


def test_transforms():
    """Test different dct transforms by embedding sequences with best layers from test_layers() and
    transforming each family's average embedding with different DCT dimensions. Sequences from the
    full pfam db are then searched against these dct representations and the results are logged.
    """

    # DCT transforms to test
    i = [3, 4, 5, 6, 7, 8]
    j = [20, 30, 40, 50, 60, 70, 80]

    # Embed using layers 17 and 23
    for lay in [17, 25]:

        # For every combination of i and j, transform embeddings from layers 17 and 25 and
        # search the full pfam db against the transformed embeddings
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


def get_seqs(fam: str, direc: str, seqs: list) -> list:
    """Takes a family name and directory and returns a list of sequences from that family where
    each sequence in the list is unique. Some sequences will have the same name but different
    regions of the protein.

    :param fam: family name
    :param dir: directory containing a singular fasta file
    :param seqs: list of sequences
    :return: list of sequences
    """

    for fam_dir in os.listdir(direc):
        if fam == fam_dir:  # Looking specifically for families missed in prior searches
            for seq in SeqIO.parse(f'{direc}/{fam_dir}/seqs.fa', 'fasta'):
                if seq.id == 'consensus':  # Skip consensus sequence
                    continue
                seq = (seq.id, str(seq.seq))
                if seq not in seqs:  # Make sure seq is unique (id can be the same)
                    seqs.append(seq)

    return seqs


def get_transforms(seqs: list, tokenizer, model, device: str) -> list:
    """Returns a list of transformed embeddings for each sequence in seqs.

    :param seqs: list of sequences in a Pfam family
    :param tokenizer: tokenizer
    :param model: model
    :param device: cpu/gpu
    """

    transforms = []
    for seq in seqs:
        embed = Embedding(seq[0], seq[1], None)  #pylint: disable=E1136
        embed.embed_seq(tokenizer, model, device, 'esm2', 17)
        embed = Transform(seq[0], embed.embed[1], None)  #pylint: disable=E1136
        embed.quant_2D(8, 80)
        if embed.trans[1] is not None:  # Embedding may be too short
            transforms.append(embed.trans[1])

    return transforms


def embed_fam(fam: str, tokenizer, model, device: str):
    """Embeds and transforms sequences from a Pfam family. Sequences are sampled from both the
    seed and full databases. The average transform is then calculated and saved to file.

    :param fam: family name
    :param tokenizer: tokenizer
    :param model: model
    :param device: device

    Before embedding make sure these these lines of code are in the main function of embed_pfam.py:

    with open('data/missed_families.txt', 'r', encoding='utf8') as f:
        families = f.read().splitlines()[1:]
    families = [f'data/full_seqs/{fam.split(",")[0]}' for fam in families]
    """

    # Get sequences from seed and full databases
    seqs = get_seqs(fam, 'data/full_seqs', [])
    seqs = get_seqs(fam, 'data/families_nogaps', seqs)

    # Randomly sample sequences - embed and transform
    logging.info('Embedding %s', fam)
    sample_size = len(seqs) if len(seqs) < 500 else 500
    seqs = sample(seqs, sample_size)
    transforms = []
    transforms = get_transforms(seqs, tokenizer, model, device)

    # Find average value for each position in all transforms
    logging.info('Averaging %s', fam)
    transforms = np.mean(transforms, axis=0, dtype=int)
    avg_dct = np.array([int(val) for val in transforms])
    avg_dct = Transform(fam, None, avg_dct)

    # Save avg transform to file
    with open(f'data/full_dct/{fam}.npy', 'wb') as emb:
        np.save(emb, avg_dct.trans)


def queue_fam(rank: int, queue: mp.Queue):
    """Goes through queue of families to embed and transform and which GPU to load models on.

    :param rank: GPU to load model on
    :param queue: queue of families to embed and transform
    """

    # Load model on GPU
    device = torch.device(f'cuda:{rank}')  #pylint: disable=E1101
    tokenizer, model = load_model('esm2', device)

    # Embed and transform each family until queue is empty
    while True:
        fam, direc = queue.get()
        if fam is None:
            break
        if f'{fam}.npy' in os.listdir(direc):  # Skip if already existing
            logging.info('Skipping %s...\n', fam)
            continue
        embed_fam(fam, tokenizer, model, device)


def test_full():
    """ Embeds and transforms sequences from families that were missed in prior searches.
    Initializes a queue 
    """

    # Create directory to store each avg transform
    direc = 'data/full_dct'
    if not os.path.exists(direc):
        os.makedirs(direc)

    # Get family names from file
    with open('data/missed_families.txt', 'r', encoding='utf8') as f:
        families = f.read().splitlines()
    families = [f'{fam.split(",")[0]}' for fam in families][1:]

    # Initialize queue/processes and call queue_fam
    queue = mp.Queue()
    processes = []
    for rank in range(NUM_GPUS):
        proc = mp.Process(target=queue_fam, args=(rank, queue))
        proc.start()
        processes.append(proc)
    for fam in families:
        queue.put((fam, direc))
    for _ in range(NUM_GPUS):
        queue.put((None, None))
    for proc in processes:
        proc.join()

    # Combine all avg transforms into one file
    avg_dcts = []
    for fam in families:
        if f'{fam}.npy' in os.listdir('data/full_dct'):
            with open(f'data/full_dct/{fam}.npy', 'rb') as emb:
                avg_dcts.append(np.load(emb, allow_pickle=True))
    with open('data/full_dct/full_dct.npy', 'wb') as emb:
        np.save(emb, avg_dcts)

    # Search against full pfam db
    os.system('python scripts/search_dct.py -d data/full_dct/full_dct.npy -l 17 -s1 8 -s2 80')


def main():
    """Main calls test functions.
    """

    test_full()


if __name__ == '__main__':
    main()
