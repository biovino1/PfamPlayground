"""================================================================================================
This embeds all of the sequences from Pfam using RostLab's ProtT5_XL_UniRef50 model. The embeddings
are saved as numpy arrays.

Ben Iovino  05/08/23   PfamPlayground
================================================================================================"""

import logging
import os
import torch
import numpy as np
import regex as re
from transformers import T5EncoderModel, T5Tokenizer

logging.basicConfig(filename='embed_pfam.log', level=logging.INFO, format='%(asctime)s %(message)s')


def prot_t5xl_embed(seq: str, tokenizer, encoder, device) -> list:
    """=============================================================================================
    This function accepts a protein sequence and returns a list of vectors, each vector representing
    a single amino acid using RostLab's ProtT5_XL_UniRef50 model.

    :param seq: protein sequence
    :param tokenizer: tokenizer model
    :param encoder: encoder model
    :param device: gpu/cpu
    return: list of vectors
    ============================================================================================="""

    # Remove special/gaps chars add space after each amino acid so each residue is vectorized
    seq = re.sub(r"[UZOB]", "X", seq).upper()
    seq = re.sub(r"\.", "", seq)
    seq = [' '.join([*seq])]

    # Tokenize, encode, and load sequence
    ids = tokenizer.batch_encode_plus(seq, add_special_tokens=True, padding=True)
    input_ids = torch.tensor(ids['input_ids']).to(device)  # pylint: disable=E1101
    attention_mask = torch.tensor(ids['attention_mask']).to(device)  # pylint: disable=E1101

    # Extract sequence features
    with torch.no_grad():
        embedding = encoder(input_ids=input_ids,attention_mask=attention_mask)
    embedding = embedding.last_hidden_state.cpu().numpy()

    # Remove padding and special tokens
    features = []
    for seq_num in range(len(embedding)):  # pylint: disable=C0200
        seq_len = (attention_mask[seq_num] == 1).sum()
        seq_emd = embedding[seq_num][:seq_len-1]
        features.append(seq_emd)
    return features[0]


def embed_fam(path: str, tokenizer, model, device):
    """=============================================================================================
    This function accepts a directory that contains fasta files of protein sequences and embeds
    each sequence using the provided tokenizer and encoder. The embeddings are saved as numpy
    arrays in a new directory.

    :param path: directory containing fasta files
    :param tokenizer: tokenizer model
    :param model: encoder model
    :param device: gpu/cpu
    ============================================================================================="""

    # Get last directory in path
    ref_dir = path.rsplit('/', maxsplit=1)[-1]
    if not os.path.isdir(f'prott5_embed/{ref_dir}'):
        os.makedirs(f'prott5_embed/{ref_dir}')

    # Get fasta files in ref_dir
    files = [f'{path}/{file}' for file in os.listdir(path) if file.endswith('.fa')]

    # Open each fasta file
    for i, file in enumerate(files):  # pylint: disable=W0612

        # Check if embedding already exists
        if os.path.exists(f'prott5_embed/{ref_dir}/{file.split("/")[-1].replace(".fa", ".txt")}'):
            logging.info('Embedding for %s already exists. Skipping...', file)
            continue

        with open(file, 'r', encoding='utf8') as fa_file:
            logging.info('Embedding %s...', file)

            # Get sequence and remove newline characters
            seq = ''.join([line.strip('\n') for line in fa_file.readlines()[1:]])

            # Embed sequence
            seq_emd = prot_t5xl_embed(seq, tokenizer, model, device)

            # Save embedding as numpy array
            filename = file.rsplit('/', maxsplit=1)[-1].replace('.fa', '.txt')
            with open(f'prott5_embed/{ref_dir}/{filename}', 'w', encoding='utf8') as f:
                np.savetxt(f, seq_emd, fmt='%4.6f', delimiter=' ')

    logging.info('Finished embedding sequences in %s\n', ref_dir)


def main():
    """=============================================================================================
    Main loads the tokenizer and encoder models and calls embed_fam to embed all sequences in each
    family directory from Pfam.
    ============================================================================================="""

    logging.info('Loading tokenizer and encoder models...\n')
    if os.path.exists('t5_tok.pt'):
        tokenizer = torch.load('t5_tok.pt')
    else:
        tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
        torch.save(tokenizer, 't5_tok.pt')
    if os.path.exists('prot_t5_xl.pt'):
        model = torch.load('prot_t5_xl.pt')
    else:
        model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
        torch.save(model, 'prot_t5_xl.pt')

    # Load model to gpu if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # pylint: disable=E1101
    model.to(device)

    # Get names of all family folders and embed all seqs in each one
    families = [f'families_nogaps/{fam}' for fam in os.listdir('families_nogaps')]
    for fam in families:
        logging.info('Embedding sequences in %s...', fam)
        embed_fam(fam, tokenizer, model, device)


if __name__ == '__main__':
    main()
