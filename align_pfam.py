"""================================================================================================
This script uses PEbA to align sequences from each family to a sample of sequences from the same
family and a sample from sequences in other families. Each alignment is saved in a directory named
after the sequence and alignment scores are saved in a csv file.

Ben Iovino  05/09/23   PfamPlayground
================================================================================================"""

import os
from random import sample


def sample_sequences(families: list, sample_size: int):
    """=============================================================================================
    This function accepts a list of directories and a sample size and returns a list of random
    sequences, one from each sampled directory.

    :param families: list of Pfam families
    :param sample_size: number of sequences to sample from each family
    return list: list of sample sequences
    ============================================================================================="""

    # Randomly sample families
    fam_samp = sample(families, sample_size)

    # Sample one sequence from each family
    samples = []
    for family in fam_samp:
        family = family.replace('Data/avg_embed/', 'Data/families_nogaps/')
        seqs = [f'{family}/{file}' for file in os.listdir(family)]
        samples.append(sample(seqs, 1)[0])
    return samples


def sample_embeddings(fam: str, families: list, sample_size: int):
    """=============================================================================================
    This function accepts a list of directories and a sample size and returns a list of random
    embeddings, one from each sampled directory, including one from the given family.

    :param fam: family to include in sample
    :param families: list of Pfam families
    :param sample_size: proportion of sequences to sample from each family
    return list: list of sample sequences
    ============================================================================================="""

    # Retrieve emb for given family and randomly sample the rest
    fam_emb = f'Data/avg_embed/{fam}'
    fam_samp = sample(families, sample_size-1)
    fam_samp.append(fam_emb)

    return fam_samp


def align_samples(samples: list, families: list):
    """=============================================================================================
    This function accepts a list of fasta files and aligns each sequence to each other using PEbA
    and BLOSUM to get the alignment score from both.

    :param samples: list of fasta files sampled from Pfam families
    :param families: list of Pfam families
    ============================================================================================="""

    # Align sample sequences to sampled embeddings
    for samp in samples:
        family = samp.split('/')[2]
        emb_samp = sample_embeddings(family, families, 1)

        # Need four files to run PEbA
        # Samp -> fasta1; families_nogaps/{family}/consensus.fa -> fasta2
        # prott5_embed/{family}/{seq}.txt -> emb1; {emb}/avg_embed.txt -> emb2
        for emb in emb_samp:
            print(emb)
            fasta1 = samp
            fasta2 = f'Data/families_nogaps/{emb.split("/")[2]}/consensus.fa'
            emb1 = f'Data/prott5_embed/{family}/{samp.split("/")[2].replace(".fa", ".txt")}'  #pylint: disable=W0612
            emb2 = f'{emb}/avg_embed.txt'

            # Create directory for alignment
            fam1, fam2 = samp.split('/')[1], emb.split('/')[1]
            seq1 = samp.split('/')[2].strip('.fa')
            direc = f'{fam1}/{seq1}-{fam2}'

            # Create directories for PEbA and BLOSUM alignments
            if not os.path.exists(f'Data/alignments/PEbA/{direc}'):
                os.makedirs(f'Data/alignments/PEbA/{direc}')
            if not os.path.exists(f'Data/alignments/blosum/{direc}'):
                os.makedirs(f'Data/alignments/blosum/{direc}')

            # Call PEbA
            os.system(f'python PEbA/peba.py -f1 {fasta1} -f2 {fasta2} '
                      f'-e2 {emb2} -s Data/alignments/PEbA/{direc}/align.msf')

            # Call BLOSUM
            os.system(f'python PEbA/local_MATRIX.py -f1 {fasta1} -f2 {fasta2} '
                      f'-sf Data/alignments/blosum/{direc}/align.msf')


def main():
    """=============================================================================================
    Main gets families from directory of Pfam embeddings and samples sequences from each family.
    The sequences are then all aligned to each other using PEbA.
    ============================================================================================="""

    # Get families from embedding directory
    emb_dir = 'Data/avg_embed'
    families = [f'{emb_dir}/{file}' for file in os.listdir(emb_dir)]

    # Get sample sequences from each family and align them to each other
    samples = sample_sequences(families, 1)
    align_samples(samples, families)


if __name__ == '__main__':
    main()
