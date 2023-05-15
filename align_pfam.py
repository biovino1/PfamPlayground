"""================================================================================================
This script uses PEbA to align sequences from each family to a sample of sequences from the same
family and a sample from sequences in other families. Each alignment is saved in a directory named
after the sequence and alignment scores are saved in a csv file.

Ben Iovino  05/09/23   PfamPlayground
================================================================================================"""


import os
from random import sample


def sample_families(families: list, sample_prop: float):
    """=============================================================================================
    This function accepts a list of directories and a sample size and returns a list of sampled
    sequences from each directory.

    :param families: list of Pfam families
    :param sample_size: proportion of sequences to sample from each family
    return list: list of sample sequences
    ============================================================================================="""

    # Sample sequences from each family
    samples = []
    for family in families:
        sample_size = round(len(os.listdir(family)) * sample_prop)
        seqs = [f'{family}/{file}' for file in os.listdir(family)]
        samples.append(sample(seqs, sample_size))
    return samples


def align_samples(samples: list):
    """=============================================================================================
    This function accepts a list of fasta files and aligns each sequence to each other using PEbA
    and BLOSUM to get the alignment score from both.

    :param samples: list of fasta files sampled from Pfam families
    ============================================================================================="""

    # Align sample sequences to each other
    for samp1 in samples:
        for samp2 in samples:
            if samp1 == samp2:  # Skip if same sample
                continue

            # Get fasta files for sample and sequences
            samp1_split, samp2_split = samp1.split('/'), samp2.split('/')
            samp1_fasta = f'families/{samp1_split[1]}/{samp1_split[2].replace(".txt", ".fa")}'
            samp2_fasta = f'families/{samp2_split[1]}/{samp2_split[2].replace(".txt", ".fa")}'

            # Create directory for alignment
            fam1, fam2 = samp1_split[1], samp2_split[1]
            seq1, seq2 = samp1_split[2].strip('.txt'), samp2_split[2].strip('.txt')
            direc = f'{fam1}/{seq1}-{fam2}'

            # Create directories for PEbA and BLOSUM alignments
            if not os.path.exists(f'alignments/PEbA/{direc}'):
                os.makedirs(f'alignments/PEbA/{direc}')
            if not os.path.exists(f'alignments/blosum/{direc}'):
                os.makedirs(f'alignments/blosum/{direc}')

            # Call PEbA
            os.system(f'python PEbA/peba.py -f1 {samp1_fasta} -f2 {samp2_fasta} '
                      f' -e1 {samp1} -e2 {samp2} -s alignments/PEbA/{direc}/{seq2}')

            # Call BLOSUM
            os.system(f'python PEbA/local_MATRIX.py -f1 {samp1_fasta} -f2 {samp2_fasta} '
                      f'-sf alignments/blosum/{direc}/{seq2}')


def main():
    """=============================================================================================
    Main gets families from directory of Pfam embeddings and samples sequences from each family.
    The sequences are then all aligned to each other using PEbA.
    ============================================================================================="""

    # Get families from embedding directory
    emb_dir = 'prott5_embed'
    families = [f'{emb_dir}/{file}' for file in os.listdir(emb_dir)]

    # Get sample sequences from each family and align them to each other
    samples = sample_families(families, 0.5)
    samples = [seq for fam in samples for seq in fam]
    align_samples(samples)


if __name__ == '__main__':
    main()
