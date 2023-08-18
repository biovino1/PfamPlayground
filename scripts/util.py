"""This script defines the embedding class, which is used to embed protein sequences using the
ProtT5_XL_UniRef50 and ESM-2_t36_3B models and transform them to DCT vectors.

__author__ = "Ben Iovino"
__date__ = "07/21/23"
"""

import re
import esm
import torch
import numpy as np
from scipy.fft import dct, idct
from transformers import T5EncoderModel, T5Tokenizer
from scipy.spatial.distance import cityblock


def load_model(encoder: str, device: str) -> tuple:
    """Loads and returns tokenizer and encoder. Outside of embedding class so it can be loaded
    once per script, much faster.

    :param encoder: prott5 or esm2
    :param device: cpu or gpu
    :return: tuple containing tokenizer and model
    """

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
    """This class stores embeddings for a single protein sequence.
    """


    def __init__(self, seqid: str, seq: str, embed: None):
        """Defines embedding class, which is a protein id, sequence, and embedding.

        :param seqid: protein ID
        :param seq: protein sequence
        :param embed: embedding (n x m matrix), not initialized by default
        """

        self.seq = np.array([seqid, seq], dtype=object)
        self.embed = np.array([seqid, embed], dtype=object)


    def clean_seq(self):
        """Returns a protein sequence after removing special characters, gaps and converting
        all characters to uppercase.

        :param self: protein ID and sequence
        """

        self.seq[1] = re.sub(r"[UZOB]", "X", self.seq[1]).upper()
        self.seq[1] = re.sub(r"\.", "", self.seq[1])  #//NOSONAR
        self.seq[1] = [' '.join([*self.seq[1]])]


    def prot_t5xl_embed(self, tokenizer, model, device: str):
        """Returns embedding of a protein sequence. Each vector represents a single amino
        acid using RostLab's ProtT5_XL_UniRef50 model.

        :param seq: protein ID and sequence
        :param model: dict containing tokenizer and encoder
        :param device: gpu/cpu
        """

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
        """Returns embedding of a protein sequence. Each vector represents a single amino
        acid using Facebook's ESM2 model.

        :param seq: protein ID and sequence
        :param tokenizer: tokenizer
        :param model: encoder model
        :param device: gpu/cpu
        return: list of vectors
        """

        # Embed sequences
        self.seq[1] = self.seq[1].upper()  # tok does not convert to uppercase
        _, _, batch_tokens = tokenizer([self.seq])
        batch_tokens = batch_tokens.to(device)  # send tokens to gpu

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[layer])
        embed = results["representations"][layer].cpu().numpy()
        self.embed[1] = embed[0]


    def embed_seq(self, tokenizer, model, device: str, encoder: str, layer: int):
        """Returns embedding of a protein sequence.

        :param tokenizer: tokenizer
        :param model: encoder model
        :param device: gpu/cpu
        :param encoder: prott5 or esm2
        :param layer: layer to extract features from (if using esm2)
        """

        # ProtT5_XL_UniRef50 or ESM-2_t36_3B
        if encoder == 'prott5':
            self.prot_t5xl_embed(tokenizer, model, device)
        if encoder == 'esm2':
            self.esm2_embed(tokenizer, model, device, layer)


    def search(self, search_db: np.ndarray, top: int) -> dict:
        """Searches embedding against a database of embeddings

        :param database: array of embeddings
        :param top: number of results to return
        :return: dict where keys are family names and values are similarity scores
        """

        sims = {}
        for embed in search_db:
            fam, embed = embed[0], embed[1]

            # np.load loads single line as 1D array, convert to 2D
            if len(embed) > 10:
                embed = [embed]

            # Find most similary embedding in query to embedding in database
            for pos1 in embed:  # db embed
                sim_list = []
                for pos2 in self.embed[1]:  # query embed
                    sim_list.append(1-cityblock(pos1, pos2))
                max_sim = max(sim_list)  # most similar position between the two
                sims[fam] = sims.get(fam, []) + [max_sim]
            sims[fam] = np.mean(sims[fam])  # overall similarity between the two

        # Sort sims dict and return top n results
        sims = dict(sorted(sims.items(), key=lambda item: item[1], reverse=True)[0:top])

        return sims


class Transform:
    """This class stores inverse discrete cosine transforms (iDCT) for a single protein sequence.
    """


    def __init__(self, seqid: str, embed: np.ndarray, transform: None):
        """Defines transform class, which is a protein id, embedding, and transform.

        :param seqid: protein ID
        :param embed: embedding (n x m matrix)
        :param transform: transform (n*m 1D array), not initialized by default
        """

        self.embed = np.array([seqid, embed], dtype=object)
        self.trans = np.array([seqid, transform], dtype=object)


    def scale(self, vec: np.ndarray) -> np.ndarray:
        """Scale from protsttools. Takes a vector and returns it scaled between 0 and 1.

        :param v: vector to be scaled
        :return: scaled vector
        """

        maxi = np.max(vec)
        mini = np.min(vec)
        return (vec - mini) / float(maxi - mini)


    def iDCT_quant(self, vec: np.ndarray, num: int) -> np.ndarray:
        """iDCTquant from protsttools. Takes a vector and returns its iDCT.

        :param vec: vector to be transformed
        :param num: number of coefficients to keep
        :return: transformed vector
        """

        f = dct(vec.T, type=2, norm='ortho')
        trans = idct(f[:,:num], type=2, norm='ortho')  #pylint: disable=E1126
        for i in range(len(trans)):  #pylint: disable=C0200
            trans[i] = self.scale(trans[i])  #pylint: disable=E1137
        return trans.T  #pylint: disable=E1101


    def quant_2D(self, n_dim: int, m_dim: int):
        """quant2D from protsttools. Takes an embedding and returns its iDCT on both axes.

        :param emb: embedding to be transformed (n x m array)
        :param n_dim: number of coefficients to keep on first axis
        :param m_dim: number of coefficients to keep on second axis
        """

        dct = self.iDCT_quant(self.embed[1][1:len(self.embed[1])-1], n_dim)  #pylint: disable=W0621
        ddct = self.iDCT_quant(dct.T, m_dim).T
        try:
            ddct = ddct.reshape(n_dim * m_dim)
            self.trans[1] = (ddct*127).astype('int8')
        except ValueError:  # If embedding is too small to transform
            self.trans[1] = None


    def concat(self, vec: np.ndarray):
        """Concatenates a vector to the transform.

        :param vec: vector to be added to transform
        """

        transform = self.trans[1]
        if transform is None:  # Initialize transform if empty
            self.trans[1] = vec
        else:
            self.trans[1] = np.concatenate((transform, vec))


    def search(self, search_db: np.ndarray, top: int) -> dict:
        """Searches transform against a database of transforms:

        :param database: array of transforms
        :param top: number of results to return
        :return: dict where keys are family names and values are similarity scores
        """

        # Search query against every dct embedding
        sims = {}
        for transform in search_db:
            fam, db_dct = transform[0], transform[1]  # family name, dct vector for family
            sims[fam] = 1-cityblock(db_dct, self.trans[1]) # compare query to dct

        # Return first n results
        sims = dict(sorted(sims.items(), key=lambda item: item[1], reverse=True)[0:top])

        return sims
