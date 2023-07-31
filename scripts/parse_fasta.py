"""================================================================================================
This script parses each sequence in the Pfam-A.full database and creates a fasta file for each
sequence in the family. The fasta files are stored in a directory named after the family.

Ben Iovino  06/05/23   SearchEmb
================================================================================================"""

import os


def write_seq(seq_id: str, fam: str, seq: str):
    """=============================================================================================
    This function accepts a sequence id, the family it belongs to, and the fasta sequence. It writes
    the sequence to a file in the directory named after the family.

    :param seq_id: sequence id
    :param fam: family name
    :param seq: fasta sequence
    ============================================================================================="""

    # Split seq_id for file name
    sid = seq_id.split('/')[0]
    region = seq_id.split('/')[1]

    # Add newline characters to seq
    seq = [seq[i:i+50] for i in range(0, len(seq), 50)]
    seq = '\n'.join(seq)

    # Write to file
    with open(f'data/full_seqs/{fam}/seqs.fa', 'a', encoding='utf8') as file:
        file.write(f'{sid}\t{region}\t{fam}\n{seq}\n')


def parse_id(line: str) -> tuple:
    """=============================================================================================
    This function accepts an id line from the pfam fasta database and returns the sequence id and
    family name.

    :param line: single line from Pfam database file
    :return tuple: sequence id and family name
    ============================================================================================="""

    line = line.split()
    seq_id = line[0]  # Seq id and region
    fam = line[2].split(';')[1]  # Take GF ID, not AC ID

    return seq_id, fam


def read_pfam(pfam: str):
    """=============================================================================================
    This function accepts a pfam database file and parses individual sequences into a file for each
    sequence in each family. The files are stored in a directory named after the family.

    :param pfam: Pfam database file
    ============================================================================================="""

    seq_id, fam, seq = '', '', ''
    with open(pfam, 'r', encoding='utf8', errors='replace') as file:
        for line in file:

            # ID line, new seq
            if line.startswith('>'):

                # Write previous seq to file
                if seq_id:
                    write_seq(seq_id, fam, seq)

                # Get new seq id and family
                seq_id, fam = parse_id(line)
                if not os.path.exists(f'data/full_seqs/{fam}'):
                    os.mkdir(f'data/full_seqs/{fam}')
                seq = ''
                continue

            # Add to seq
            seq += line.strip()


def main():
    """=============================================================================================
    Main detects if Pfam-A.full database is in directory. If not, it will download from Pfam
    website and unzip. Then, it will call read_pfam to parse each family into individual fasta
    files.
    ============================================================================================="""

    # Read Pfam-A.fasta if it exists
    pfam_full = 'data/Pfam-A.fasta'
    if not os.path.exists(pfam_full):
        print('Pfam-A.seed not found. Downloading from Pfam...')
        if not os.path.exists('data'):
            os.mkdir('data')
        os.system('wget -P data ' \
            'https://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam35.0/Pfam-A.fasta.gz')
        os.system('gunzip data/Pfam-A.fasta.gz')

    # Create directories for families
    if not os.path.exists('data/full_seqs'):
        os.mkdir('data/full_seqs')

    # Parse all fasta seqs
    read_pfam(pfam_full)


if __name__ == '__main__':
    main()
