"""================================================================================================
This script contains utility functions to be imported into other scripts.

Ben Iovino  06/01/23   SearchEmb
================================================================================================"""

import argparse
import re
import esm
import torch
from transformers import T5EncoderModel, T5Tokenizer


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


def embed_seq(seq: tuple, tokenizer, model, device: str, args: argparse.Namespace) -> list:
    """=============================================================================================
    This function accepts a protein sequence and returns a list of vectors, each vector representing
    a single amino acid using the provided tokenizer and encoder.

    :param seq: protein ID and sequence
    :param tokenizer: tokenizer
    :param model: encoder model
    :param device: gpu/cpu
    :param args: encoder type and layer to extract features from (if using esm2)
    return embed: list of vectors
    ============================================================================================="""

    # ProtT5_XL_UniRef50
    if args.e == 'prott5':
        embed = prot_t5xl_embed(seq, tokenizer, model, device)

    # ESM-2_t36_3B
    elif args.e == 'esm2':
        embed = esm2_embed(seq, tokenizer, model, device, args.l)

    return embed


def prot_t5xl_embed(seq: tuple, tokenizer, model, device) -> list:
    """=============================================================================================
    This function accepts a protein sequence and returns a list of vectors, each vector representing
    a single amino acid using RostLab's ProtT5_XL_UniRef50 model.

    :param seq: protein ID and sequence
    :param model: dict containing tokenizer and encoder
    :param device: gpu/cpu
    return: list of vectors
    ============================================================================================="""

    # Tokenize, encode, and load sequence
    seq = clean_seq(seq[1])
    ids = tokenizer.batch_encode_plus(seq, add_special_tokens=True, padding=True)
    input_ids = torch.tensor(ids['input_ids']).to(device)  # pylint: disable=E1101
    attention_mask = torch.tensor(ids['attention_mask']).to(device)  # pylint: disable=E1101

    # Extract sequence features
    with torch.no_grad():
        embedding = model(input_ids=input_ids,attention_mask=attention_mask)
    embedding = embedding.last_hidden_state.cpu().numpy()  # pylint: disable=E1101

    # Remove padding and special tokens
    features = []
    for seq_num in range(len(embedding)):  # pylint: disable=C0200
        seq_len = (attention_mask[seq_num] == 1).sum()
        seq_emd = embedding[seq_num][:seq_len-1]
        features.append(seq_emd)

    return features[0]


def esm2_embed(seq: tuple, tokenizer, model, device: str, layer: int) -> list:
    """=============================================================================================
    This function accepts a protein sequence and returns a list of vectors, each vector representing
    a single amino acid using Facebook's ESM-2 model.

    :param seq: protein ID and sequence
    :param tokenizer: tokenizer
    :param model: encoder model
    :param device: gpu/cpu
    return: list of vectors
    ============================================================================================="""

    # Embed sequences
    seq = (seq[0], seq[1].upper())  # tok does not convert to uppercase
    batch_labels, batch_strs, batch_tokens = tokenizer([seq])  #pylint: disable=W0612
    batch_tokens = batch_tokens.to(device)  # send tokens to gpu

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[layer])
    embed = results["representations"][layer].cpu().numpy()

    return embed[0]


def load_model(encoder: str, device: str) -> tuple:
    """=============================================================================================
    This function loads the ProtT5-XL model and tokenizer and returns them.

    :param encoder: prott5 or esm2
    :param device: cpu or gpu
    :return tuple: tokenizer and model
    ============================================================================================="""

    # ProtT5_XL_UniRef50
    if encoder == 'prott5':
        tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50', do_lower_case=False)
        model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
        model.to(device)  # Loads to GPU if available

    # ESM-2_t36_3B
    if encoder == 'esm2':
        model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        tokenizer = alphabet.get_batch_converter()
        model.eval()  # disables dropout for deterministic results
        model.to(device)

    return tokenizer, model
