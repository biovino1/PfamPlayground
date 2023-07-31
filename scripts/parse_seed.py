"""================================================================================================
This script parses each family in the Pfam-A.seed database and creates a fasta file for each
sequence in the family. The fasta files are stored in a directory named after the family.

Ben Iovino  05/01/23   SearchEmb
================================================================================================"""

import os
import regex as re


def clean_fasta(seq: str, cons: bool, gaps: bool) -> str:
    """=============================================================================================
    This function accepts a fasta sequence with gaps and returns it in fasta format (newline char
    every 50 characters).

    :param seq: fasta sequence
    :param cons: flag to indicate if sequence is consensus sequence
    :param gaps: flag to indicate if gaps should be included in sequences
    :return str: fasta sequence with newline characters
    ============================================================================================="""

    if gaps is True:  # Gaps included
        if cons is False:  # All other sequences
            seq = '\n'.join(seq[i:i+50] for i in range(0, len(seq), 50))
        else:  # Consensus sequence
            seq = re.sub(r'[\+\-]', 'X', seq)  # replace +/- with X for embedding purposes
            seq = '\n'.join(seq[i:i+50] for i in range(0, len(seq), 50))

    else:  # Gaps are not included
        if cons is False:  # All other sequences
            seq = re.sub(r'\.', '', seq)
            seq = '\n'.join(seq[i:i+50] for i in range(0, len(seq), 50))
        else:  # Consensus sequence
            seq = re.sub(r'[\+\-]', 'X', seq)  # replace +/- with X for embedding purposes
            seq = re.sub(r'\.', '', seq)
            seq = '\n'.join(seq[i:i+50] for i in range(0, len(seq), 50))

    return seq


def write_fasta(family: str, seqs: list, gaps: bool, fam_dir: str):
    """=============================================================================================
    This function accepts a list of lines from the pfam database that contain sequences in the same
    family and writes them to the same file.

    :param family: name of family that line is from
    :param list: list of lines from pfam database
    :param gaps: flag to indicate if gaps should be included in sequences
    :param fam_dir: directory to store families
    ============================================================================================="""

    with open(f'{fam_dir}/{family}/seqs.fa', 'w', encoding='utf8') as file:
        for line in seqs:

            # Consenus sequence written slightly differently than other sequences
            if line.startswith('#=GC seq_cons'):
                line = line.split()
                seq = clean_fasta(line[2], True, gaps)
                length = len(seq.replace('\n', ''))-seq.count('.')
                file.write(f'>consensus\tlength {length}\n{seq}')

            else:  # All other sequences

                # Isolate AC number and region
                line = line.split()
                seq_id, region = line[0].split('/')[0], line[0].split('/')[1]
                seq = clean_fasta(line[1], False, gaps)
                file.write(f'>{seq_id}\t{region}\t{family}\n{seq}\n')


def read_pfam(pfam: str, gaps: bool, fam_dir: str):  #\\NOSONAR
    """=============================================================================================
    This function accepts a pfam database file and parses individual sequence into a file for each
    sequence in each family. The files are stored in a directory named after the family.

    :param pfam: Pfam database file
    :param gaps: flag to indicate if gaps should be included in sequences
    :param fam_dir: name of directory to store families
    ============================================================================================="""

    with open(pfam, 'r', encoding='utf8', errors='replace') as file:
        in_fam, seqs = False, []  # Flag to indicate if we are in a family and list to store seqs
        for line in file:

            # Read until you reach #=GF ID
            if line.startswith('#=GF ID'):
                family = line.split()[2]
                in_fam = True

                # Create a directory for the family
                if not os.path.exists(family):
                    os.mkdir(f'{fam_dir}/{family}')

            # If in a family, read sequences and write each one to a file
            elif in_fam:
                if line.startswith('#'):  # Skip GR, GF, GS lines
                    if line.startswith('#=GC seq_cons'):  # Except for consensus seq
                        seqs.append(line)
                        continue
                    continue
                if line.startswith('//'):  # End of family
                    write_fasta(family, seqs, gaps, fam_dir)
                    in_fam, seqs = False, []
                    continue
                seqs.append(line)


def main():
    """=============================================================================================
    Main detects if Pfam-A.seed database is in directory. If not, it will download from Pfam
    website and unzip. Then, it will call read_pfam to parse each family into individual fasta
    files.
    ============================================================================================="""

    # Read Pfam-A.seed if it exists
    pfam_seed = 'data/Pfam-A.seed'
    if not os.path.exists('data/Pfam-A.seed'):
        print('Pfam-A.seed not found. Downloading from Pfam...')
        if not os.path.exists('data'):
            os.mkdir('data')
        os.system('wget -P data ' \
             'https://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam35.0/Pfam-A.seed.gz')
        os.system('gunzip data/Pfam-A.seed.gz')

    # Create directories for families
    fam_gaps, fam_nogaps = 'data/families_gaps', 'data/families_nogaps'
    if not os.path.exists(fam_gaps):
        os.mkdir(fam_gaps)
    if not os.path.exists(fam_nogaps):
        os.mkdir(fam_nogaps)

    # Parse once with gaps and once without
    for gaps in [True, False]:
        if gaps is True:
            fam_dir = fam_gaps
        else:
            fam_dir = fam_nogaps
        read_pfam(pfam_seed, gaps, fam_dir)


if __name__ == "__main__":
    main()
