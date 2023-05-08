"""================================================================================================
This script parses each family in the Pfam-A.seed database and creates a fasta file for each
sequence in the family. The fasta files are stored in a directory named after the family.

Ben Iovino  05/01/23   PfamPlayground
================================================================================================"""

import os


def clean_fasta(seq: str):
    """=============================================================================================
    This function accepts a fasta sequence with gaps and returns it with no gaps and split every
    50 characters (.fa format).

    :param seq: fasta sequence with gaps
    :return str: fasta sequence without gaps
    ============================================================================================="""

    seq = seq.replace('.', '')
    seq = '\n'.join(seq[i:i+50] for i in range(0, len(seq), 50))
    return seq


def read_pfam(pfam: str):
    """=============================================================================================
    This function accepts a pfam database file and parses individual sequence into a file for each
    sequence in each family. The files are stored in a directory named after the family.

    :param pfam: Pfam database file
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
                    os.mkdir(f'families/{family}')

            # If in a family, read sequences and write each one to a file
            elif in_fam:
                if line.startswith('#'):  # Skip GR, GF, GS lines
                    continue
                if line.startswith('//'):  # End of family
                    in_fam = False
                    continue

                # Isolate AC number and region
                line = line.split()
                seq_id, region = line[0].split('/')[0], line[0].split('/')[1]
                seq = clean_fasta(line[1])
                with open(f'families/{family}/{seq_id}.fa', 'w', encoding='utf8') as f:
                    f.write(f'>{seq_id}\t{region}\n{seq}')


def main():

    # Check if directory for families exist
    if not os.path.exists('families'):
        os.mkdir('families')

    # Read Pfam-A.seed if it exists
    if os.path.exists('Pfam-A.seed'):
        pfam = 'Pfam-A.seed'
    else:
        print('Pfam-A.seed not found. Downloading from Pfam...')
        os.system('wget https://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam35.0/Pfam-A.seed.gz')
        os.system('unzip Pfam-A.seed.gz')
        pfam = 'Pfam-A.seed'
    read_pfam(pfam)


if __name__ == "__main__":
    main()
