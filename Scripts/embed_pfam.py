"""================================================================================================
This embeds all of the sequences from Pfam using RostLab's ProtT5_XL_UniRef50 model. The embeddings
are saved as numpy arrays.

Ben Iovino  05/08/23   SearchEmb
================================================================================================"""

import argparse
import logging
import os
import torch
import numpy as np
from utility import prot_t5xl_embed, esm2_embed, load_model

logging.basicConfig(filename='Data/embed_pfam.log',
                     level=logging.INFO, format='%(asctime)s %(message)s')


def embed_fam(path: str, tokenizer, model, device, encoder: str):
    """=============================================================================================
    This function accepts a directory that contains fasta files of protein sequences and embeds
    each sequence using the provided tokenizer and encoder. The embeddings are saved as numpy
    arrays in a new directory.

    :param path: directory containing fasta files
    :param tokenizer: tokenizer model
    :param model: encoder model
    :param device: gpu/cpu
    :param encoder: name of encoder
    ============================================================================================="""

    # Get last directory in path
    ref_dir = path.rsplit('/', maxsplit=1)[-1]
    if not os.path.isdir(f'Data/{encoder}_embed/{ref_dir}'):
        os.makedirs(f'Data/{encoder}_embed/{ref_dir}')

    # Get fasta files in ref_dir
    files = [f'{path}/{file}' for file in os.listdir(path) if file.endswith('.fa')]

    # Open each fasta file
    for i, file in enumerate(files):  # pylint: disable=W0612

        # Check if embedding already exists
        if os.path.exists(f'Data/{encoder}_embed/{ref_dir}/'
                        f'{file.split("/")[-1].replace(".fa", ".txt")}'):
            logging.info('Embedding for %s already exists. Skipping...', file)
            continue

        # Get sequence, embed, and save as np binary
        with open(file, 'r', encoding='utf8') as fa_file:
            logging.info('Embedding %s...', file)
            seq = ''.join([line.strip('\n') for line in fa_file.readlines()[1:]])
            if encoder == 'prott5':
                seq_emd = prot_t5xl_embed(seq, tokenizer, model, device)
            elif encoder == 'esm2':
                seq_emd = esm2_embed(seq, tokenizer, model)
            filename = file.rsplit('/', maxsplit=1)[-1].replace('.fa', '.npy')
            with open(f'Data/{encoder}_embed/{ref_dir}/{filename}', 'wb') as emb_f:
                np.save(emb_f, seq_emd)

    logging.info('Finished embedding sequences in %s\n', ref_dir)


def main():
    """=============================================================================================
    Main loads the tokenizer and encoder models and calls embed_fam() to embed all sequences in each
    family directory from Pfam.
    ============================================================================================="""

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', type=str, default='prott5')
    args = parser.parse_args()

    # Load tokenizer and encoder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #pylint: disable=E1101
    tokenizer, model = load_model(args.e, device)

    # Get names of all family folders and embed all seqs in each one
    families = [f'Data/families_nogaps/{fam}' for fam in os.listdir('Data/families_nogaps')]
    for fam in families:

        logging.info('Embedding sequences in %s...', fam)
        embed_fam(fam, tokenizer, model, device, args.e)


if __name__ == '__main__':
    main()
