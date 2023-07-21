"""================================================================================================
This script defines the embedding class, which is used to embed protein sequences using the
ProtT5_XL_UniRef50 and ESM-2_t36_3B models and transform them to DCT vectors.

Ben Iovino  07/21/23   SearchEmb
================================================================================================"""

import argparse
import re
import esm
import torch
import numpy as np
from transformers import T5EncoderModel, T5Tokenizer


def load_model(encoder: str, device: str) -> tuple:
    """=============================================================================================
    This function loads the desired encoder model and tokenizer and returns them. Outside of
    embedding class so it can be loaded once per script, much faster.

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


class Embedding:
    """============================================================================================
    This class stores embeddings and for a single protein sequence.
    ============================================================================================"""


    def __init__(self, seqid: str, seq: str, embed: None):
        """=========================================================================================
        Defines embedding class, which is a protein id, sequence, and embedding.

        :param seqid: protein ID
        :param seq: protein sequence
        :param embed: embedding (n x m matrix), not initialized by default
        ========================================================================================="""

        self.seq = np.array([seqid, seq], dtype=object)
        self.embed = np.array([seqid, embed], dtype=object)


    def clean_seq(self):
        """=========================================================================================
        This function accepts a protein sequence and returns it after removing special characters,
        gaps and converting all characters to uppercase.

        :param self: protein sequence
        :return self: protein sequence
        ========================================================================================="""

        self.seq[1] = re.sub(r"[UZOB]", "X", self.seq[1]).upper()
        self.seq[1] = re.sub(r"\.", "", self.seq[1])
        self.seq[1] = [' '.join([*self.seq[1]])]


    def prot_t5xl_embed(self, tokenizer, model, device):
        """=========================================================================================
        This function accepts a protein sequence and returns its embedding, each vector representing
        a single amino acid using RostLab's ProtT5_XL_UniRef50 model.

        :param seq: protein ID and sequence
        :param model: dict containing tokenizer and encoder
        :param device: gpu/cpu
        ========================================================================================="""

        # Tokenize, encode, and load sequence
        self.clean_seq()
        ids = tokenizer.batch_encode_plus(self.seq[1], add_special_tokens=True, padding=True)
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
        self.embed[1] = features[0]


    def esm2_embed(self, tokenizer, model, device: str, layer: int):
        """=========================================================================================
        This function accepts a protein sequence and returns a list of vectors, each vector
        representing a single amino acid using Facebook's ESM-2 model.

        :param seq: protein ID and sequence
        :param tokenizer: tokenizer
        :param model: encoder model
        :param device: gpu/cpu
        return: list of vectors
        ========================================================================================="""

        # Embed sequences
        self.seq[1] = self.seq[1].upper()  # tok does not convert to uppercase
        batch_labels, batch_strs, batch_tokens = tokenizer([self.seq])  #pylint: disable=W0612
        batch_tokens = batch_tokens.to(device)  # send tokens to gpu

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[layer])
        embed = results["representations"][layer].cpu().numpy()
        self.embed[1] = embed[0]


    def embed_seq(self, tokenizer, model, device: str, args: argparse.Namespace):
        """=========================================================================================
        This function accepts a protein id and its sequence and returns its embedding, each vector
        representing a single amino acid using the provided tokenizer and encoder.

        :param tokenizer: tokenizer
        :param model: encoder model
        :param device: gpu/cpu
        :param args: encoder type and layer to extract features from (if using esm2)
        ========================================================================================="""

        # ProtT5_XL_UniRef50 or ESM-2_t36_3B
        if args.e == 'prott5':
            self.prot_t5xl_embed(tokenizer, model, device)
        if args.e == 'esm2':
            self.esm2_embed(tokenizer, model, device, args.l)
