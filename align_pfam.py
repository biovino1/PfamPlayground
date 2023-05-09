"""================================================================================================
This script uses PEbA to align sequences from each family to a sample of sequences from the same
family and a sample from sequences in other families. Each alignment is saved in a directory named
after the sequence and alignment scores are saved in a csv file.

Ben Iovino  05/09/23   PfamPlayground
================================================================================================"""


import os
from random import sample

def main():

    # Get families from embedding directory
    emb_dir = 'prott5_embed'
    families = [f'{emb_dir}/{file}' for file in os.listdir(emb_dir)]

    # Get sample sequences from each family
    samp_size = 1
    samples = []
    for family in families:
        seqs = [f'{family}/{file}' for file in os.listdir(family)]
        samples.append(sample(seqs, samp_size))

    # Align each sample sequence to a sample of sequences from same family and other families
    for family in families:
        seqs = [f'{family}/{file}' for file in os.listdir(family)]
        
        # Align each sample to a sample of sequences in this family



if __name__ == '__main__':
    main()
