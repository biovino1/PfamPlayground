"""================================================================================================
This script parses each family in the Pfam-A.seed database and creates a fasta file for each
sequence in the family. The fasta files are stored in a directory named after the family.

Ben Iovino  05/01/23   PfamPlayground
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


def write_fasta(family: str, line: str, gaps: bool, fam_dir: str):
    """=============================================================================================
    This function accepts a line from the pfam database and writes it to a fasta file.

    :param family: name of family that line is from
    :param line: line from pfam database
    :param gaps: flag to indicate if gaps should be included in sequences
    :param fam_dir: name of directory to store families
    ============================================================================================="""

    # Consenus sequence written slightly differently than other sequences
    if line.startswith('#=GC seq_cons'):
        line = line.split()
        seq = clean_fasta(line[2], True, gaps)
        length = len(seq.replace('\n', ''))-seq.count('.')
        with open(f'{fam_dir}/{family}/consensus.fa', 'w', encoding='utf8') as file:
            file.write(f'>consensus\tlength {length}\n{seq}')

    else:  # All other sequences

        # Isolate AC number and region
        line = line.split()
        seq_id, region = line[0].split('/')[0], line[0].split('/')[1]
        seq = clean_fasta(line[1], False, gaps)
        with open(f'{fam_dir}/{family}/{seq_id}.fa', 'w', encoding='utf8') as file:
            file.write(f'>{seq_id}\t{region}\n{seq}')


def read_pfam(pfam: str, gaps: bool, fam_dir: str):
    """=============================================================================================
    This function accepts a pfam database file and parses individual sequence into a file for each
    sequence in each family. The files are stored in a directory named after the family.

    :param pfam: Pfam database file
    :param gaps: flag to indicate if gaps should be included in sequences
    :param fam_dir: name of directory to store families
    ============================================================================================="""

    with open(pfam, 'r', encoding='utf8', errors='replace') as file:
        in_fam = False  # Flag to indicate if we are in a family
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
                        write_fasta(family, line, gaps, fam_dir)
                    else: continue
                if line.startswith('//'):  # End of family
                    in_fam = False
                    continue
                write_fasta(family, line, gaps, fam_dir)  # Write sequence to file


def main():
    """=============================================================================================
    Main detects if Pfam-A.seed database is in directory. If not, it will download from Pfam
    website and unzip. Then, it will call read_pfam to parse each family into individual fasta
    files.
    ============================================================================================="""

    # Read Pfam-A.seed if it exists
    if os.path.exists('Data/Pfam-A.seed'):
        pfam = 'Data/Pfam-A.seed'
    else:
        print('Pfam-A.seed not found. Downloading from Pfam...')
        if not os.path.exists('Data'):
            os.mkdir('Data')
        os.system('wget -P Data https://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam35.0/Pfam-A.seed.gz')
        os.system('gunzip Data/Pfam-A.seed.gz')
        pfam = 'Data/Pfam-A.seed'

    # Create directories for families
    if not os.path.exists('Data/families_gaps'):
        os.mkdir('Data/families_gaps')
    if not os.path.exists('Data/families_nogaps'):
        os.mkdir('Data/families_nogaps')

    # Parse once with gaps and once without
    for gaps in [True, False]:
        if gaps is True:
            fam_dir = 'Data/families_gaps'
        else:
            fam_dir = 'Data/families_nogaps'
        read_pfam(pfam, gaps, fam_dir)


if __name__ == "__main__":
    main()
