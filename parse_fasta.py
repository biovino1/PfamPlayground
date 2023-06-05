"""================================================================================================
This script parses each sequence in the Pfam-A.full database and creates a fasta file for each
sequence in the family. The fasta files are stored in a directory named after the family.

Ben Iovino  06/05/23   PfamPlayground
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
    filename = id.replace('>','')
    region = seq_id.split('/')[1]

    # Add newline characters to seq
    seq = [seq[i:i+50] for i in range(0, len(seq), 50)]
    seq = '\n'.join(seq)

    # Write to file
    with open(f'Data/full_seqs/{fam}/{filename}.fa', 'w', encoding='utf8') as file:
        file.write(f'>{sid}\t{region}\n{seq}')



def parse_id(line: str) -> str:
    """=============================================================================================
    This function accepts an id line from the pfam fasta database and returns the sequence id and
    family name.

    :param line: single line from Pfam database file
    :return: sequence id and family name
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
                if not os.path.exists(f'Data/full_seqs/{fam}'):
                    os.mkdir(f'Data/full_seqs/{fam}')
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

    # Read Pfam-A.seed if it exists
    if os.path.exists('Data/Pfam-A.fasta'):
        pfam = 'Data/Pfam-A.fasta'
    else:
        print('Pfam-A.seed not found. Downloading from Pfam...')
        if not os.path.exists('Data'):
            os.mkdir('Data')
        os.system('wget -P Data https://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam35.0/Pfam-A.fasta.gz')
        os.system('gunzip Data/Pfam-A.fasta.gz')
        pfam = 'Data/Pfam-A.fasta'

    # Create directories for families
    if not os.path.exists('Data/full_seqs'):
        os.mkdir('Data/full_seqs')

    # Parse all fasta seqs
    read_pfam(pfam)


if __name__ == '__main__':
    main()
