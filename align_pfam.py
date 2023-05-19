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
        family = family.replace('avg_embed/', 'families_nogaps/')
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
    fam_emb = f'avg_embed/{fam}'
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
        family = samp.split('/')[1]
        emb_samp = sample_embeddings(family, families, 100)

        # Need four files to run PEbA
        # Samp -> fasta1; families_nogaps/{family}/consensus.fa -> fasta2
        # prott5_embed/{family}/{seq}.txt -> emb1; {emb}/avg_embed.txt -> emb2
        for emb in emb_samp:
            fasta1 = samp
            fasta2 = f'families_nogaps/{emb.split("/")[1]}/consensus.fa'
            emb1 = f'prott5_embed/{family}/{samp.split("/")[2].replace(".fa", ".txt")}'  #pylint: disable=W0612
            emb2 = f'{emb}/avg_embed.txt'

            # Create directory for alignment
            fam1, fam2 = samp.split('/')[1], emb.split('/')[1]
            seq1 = samp.split('/')[2].strip('.fa')
            direc = f'{fam1}/{seq1}-{fam2}'
            
            # Create directories for PEbA and BLOSUM alignments
            if not os.path.exists(f'alignments/PEbA/{direc}'):
                os.makedirs(f'alignments/PEbA/{direc}')
            if not os.path.exists(f'alignments/blosum/{direc}'):
                os.makedirs(f'alignments/blosum/{direc}')

            # Call PEbA
            os.system(f'python PEbA/peba.py -f1 {fasta1} -f2 {fasta2} '
                      f'-e2 {emb2} -s alignments/PEbA/{direc}/align.msf')

            # Call BLOSUM
            os.system(f'python PEbA/local_MATRIX.py -f1 {fasta1} -f2 {fasta2} '
                      f'-sf alignments/blosum/{direc}/align.msf')


def align_samples2(samples: list):
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
            samp1_fasta = (f'families_nogaps/{samp1_split[1]}/{samp1_split[2].replace(".txt", ".fa")}')
            samp2_fasta = (f'families_nogaps/{samp2_split[1]}/{samp2_split[2].replace(".txt", ".fa")}')

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
                      f'-e1 {samp1} -e2 {samp2} -s alignments/PEbA/{direc}/{seq2}')

            # Call BLOSUM
            os.system(f'python PEbA/local_MATRIX.py -f1 {samp1_fasta} -f2 {samp2_fasta} '
                      f'-sf alignments/blosum/{direc}/{seq2}')


def main():
    """=============================================================================================
    Main gets families from directory of Pfam embeddings and samples sequences from each family.
    The sequences are then all aligned to each other using PEbA.
    ============================================================================================="""

    # Get families from embedding directory
    emb_dir = 'avg_embed'
    families = [f'{emb_dir}/{file}' for file in os.listdir(emb_dir)]

    # Get sample sequences from each family and align them to each other
    samples = sample_sequences(families, 100)
    align_samples(samples, families)


if __name__ == '__main__':
    main()
