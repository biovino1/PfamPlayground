"""================================================================================================
This script contains utility functions to be imported into other scripts.

Ben Iovino  06/01/23   SearchEmb
================================================================================================"""

import os
import re
import torch
from transformers import T5EncoderModel, T5Tokenizer, AutoTokenizer, EsmModel


def clean_seq(seq: str) -> str:
    """=============================================================================================
    This function accepts a protein sequence and returns it after removing special characters, gaps
    and converting all characters to uppercase.

    :param seq: protein sequence
    :return seq: protein sequence
    ============================================================================================="""

    seq = re.sub(r"[UZOB]", "X", seq).upper()
    seq = re.sub(r"\.", "", seq)
    seq = [' '.join([*seq])]

    return seq


def embed_seq(seq: str, tokenizer, encoder, device, encoder_name: str) -> list:
    """=============================================================================================
    This function accepts a protein sequence and returns a list of vectors, each vector representing
    a single amino acid using the provided tokenizer and encoder.

    :param seq: protein sequence
    :param tokenizer: tokenizer model
    :param encoder: encoder model
    :param device: gpu/cpu
    :param encoder_name: name of encoder
    return seq_emb: list of vectors
    ============================================================================================="""

    # ProtT5_XL_UniRef50
    if encoder_name == 'prott5':
        embed = prot_t5xl_embed(seq, tokenizer, encoder, device)

    # ESM-2_t36_3B
    elif encoder_name == 'esm2':
        embed = esm2_embed(seq, tokenizer, encoder, device)

    return embed


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

    # Tokenize, encode, and load sequence
    seq = clean_seq(seq)
    ids = tokenizer.batch_encode_plus(seq, add_special_tokens=True, padding=True)
    input_ids = torch.tensor(ids['input_ids']).to(device)  # pylint: disable=E1101
    attention_mask = torch.tensor(ids['attention_mask']).to(device)  # pylint: disable=E1101

    # Extract sequence features
    with torch.no_grad():
        embedding = encoder(input_ids=input_ids,attention_mask=attention_mask)
    embedding = embedding.last_hidden_state.cpu().numpy()  # pylint: disable=E1101

    # Remove padding and special tokens
    features = []
    for seq_num in range(len(embedding)):  # pylint: disable=C0200
        seq_len = (attention_mask[seq_num] == 1).sum()
        seq_emd = embedding[seq_num][:seq_len-1]
        features.append(seq_emd)

    return features[0]


def esm2_embed(seq: str, tokenizer, encoder, device):
    """=============================================================================================
    This function accepts a protein sequence and returns a list of vectors, each vector representing
    a single amino acid using Facebook's ESM-2 model.

    :param seq: protein sequence
    :param: tokenizer: tokenizer model
    :param encoder: encoder model
    return: list of vectors
    ============================================================================================="""

    # Tokenize, encode, and load sequence
    seq = clean_seq(seq)
    inputs = tokenizer(seq, return_tensors="pt")
    input_ids = torch.tensor(inputs['input_ids']).to(device)  # pylint: disable=E1101
    attention_mask = torch.tensor(inputs['attention_mask']).to(device)  # pylint: disable=E1101

    # Extract sequence features
    with torch.no_grad():
        outputs = encoder(input_ids=input_ids,attention_mask=attention_mask)
    last_hidden_states = outputs.last_hidden_state.cpu().numpy()  # pylint: disable=E1101

    return last_hidden_states[0][1:-1]  # First and last tokens are BOS and EOS tokens


def load_model(encoder: str, device: str) -> tuple:
    """=============================================================================================
    This function loads the ProtT5-XL model and tokenizer and returns them.

    :param encoder: prott5 or esm2
    :param device: cpu or gpu
    :return tuple: tokenizer and model
    ============================================================================================="""

    # ProtT5_XL_UniRef50
    if encoder == 'prott5':
        if os.path.exists('Data/t5_tok.pt'):
            tokenizer = torch.load('Data/t5_tok.pt')
        else:
            tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50",
                                                     do_lower_case=False)
            torch.save(tokenizer, 'Data/t5_tok.pt')
        if os.path.exists('Data/prot_t5_xl.pt'):
            model = torch.load('Data/prot_t5_xl.pt')
        else:
            model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
            torch.save(model, 'Data/prot_t5_xl.pt')

    # ESM-2_t36_3B
    if encoder == 'esm2':
        if os.path.exists('Data/auto_tok.pt'):
            tokenizer = torch.load('Data/auto_tok.pt')
        else:
            tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
            torch.save(tokenizer, 'Data/auto_tok.pt')
        if os.path.exists('Data/esm2_t36_3B.pt'):
            model = torch.load('Data/esm2_t36_3B.pt')
        else:
            model = EsmModel.from_pretrained("facebook/esm2_t36_3B_UR50D")
            torch.save(model, 'Data/esm2_t36_3B.pt')

    # Load model to gpu if available
    model.to(device)

    return tokenizer, model
