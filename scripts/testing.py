"""================================================================================================
This script is used for various testing purposes.

Ben Iovino  07/20/23   DCTDomain
================================================================================================"""

import logging
import os
from random import sample
import numpy as np
import torch
from Bio import SeqIO
from util import load_model, Embedding, Transform


log_filename = 'data/logs/testing.log'  #pylint: disable=C0103
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
logging.basicConfig(filename=log_filename, filemode='w',
                     level=logging.INFO, format='%(message)s')


def test_layers():
    """=============================================================================================
    Test each layer of ESM2-t36-3B by embedding sequences from families that were missed in prior
    searches and transforming each family's average embedding with DCT dimensions of 5x44. Sequences
    from the full pfam db are then searched against these dct representations and the results are
    logged.
    ============================================================================================="""

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
    """=============================================================================================
    Test different dct transforms by embedding sequences with best layers from test_layers() and
    transforming each family's average embedding with different DCT dimensions. Sequences from the
    full pfam db are then searched against these dct representations and the results are logged.
    ============================================================================================="""

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


def test_full():  #\\NOSONAR
    """=============================================================================================
    Embed sequences from families that were missed in prior searches using every sequence in pfam
    full database. Because there is no alignment for the full database, we will transform each
    embedding then average the transforms for each family. These will be used as the family
    representation during search.

    Before embedding make sure these these lines of code are in the main function of embed_pfam.py:

    with open('data/missed_families.txt', 'r', encoding='utf8') as f:
        families = f.read().splitlines()[1:]
    families = [f'data/full_seqs/{fam.split(",")[0]}' for fam in families]
    ============================================================================================="""

    # Load tokenizer and encoder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #pylint: disable=E1101
    tokenizer, model = load_model('esm2', device)

    # Get family names from file
    with open('data/missed_families.txt', 'r', encoding='utf8') as f:
        families = f.read().splitlines()
    families = [f'{fam.split(",")[0]}' for fam in families][1:]

    # Read sequences from seed and full databases
    dcts = []
    for fam in families:
        seqs = []
        for fam_dir in os.listdir('data/full_seqs'):
            if fam == fam_dir:
                for seq in SeqIO.parse(f'data/full_seqs/{fam_dir}/seqs.fa', 'fasta'):
                    seqs.append((seq.id, str(seq.seq)))
        seq_list = [seq[0] for seq in seqs]
        for fam_dir in os.listdir('data/families_nogaps'):
            if fam == fam_dir:
                for seq in SeqIO.parse(f'data/families_nogaps/{fam_dir}/seqs.fa', 'fasta'):
                    if seq.id not in seq_list:  # Add only if not already in list
                        seqs.append((seq.id, str(seq.seq)))

        # Randomly sample sequences - embed and transform
        sample_size = len(seqs) if len(seqs) < 500 else 500
        seqs = sample(seqs, sample_size)
        transforms = []
        for seq in seqs:
            embed = Embedding(seq[0], seq[1], None)  #pylint: disable=E1136
            embed.embed_seq(tokenizer, model, device, 'esm2', 17)
            embed = Transform(seq[0], embed.embed[1], None)  #pylint: disable=E1136
            embed.quant_2D(8, 80)
            transforms.append(embed.trans[1])

        # Find average value for each position in all transforms
        transforms = np.mean(transforms, axis=0)
        avg_dct = np.array([int(val) for val in transforms])
        dcts.append(Transform(fam, None, avg_dct).trans)

    # Save transforms to file
    with open('data/full_dct.npy', 'wb') as emb:
        np.save(emb, dcts)

    # Search against full_dct.npy
    os.system('python scripts/search_dct.py -d data/full_dct.npy -l 17 -s1 8 -s2 80')


def main():
    """=============================================================================================
    Main function
    ============================================================================================="""

    test_full()


if __name__ == '__main__':
    main()
