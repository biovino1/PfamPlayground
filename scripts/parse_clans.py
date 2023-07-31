"""================================================================================================
This script parses each sequence in the Pfam-A.seed database and creates a fasta file for each
sequence in the family. The fasta files are stored in a directory named after the family.

Ben Iovino  06/05/23   SearchEmb
================================================================================================"""

import csv
import pickle
import os


def read_pfam(pfam: str):
    """=============================================================================================
    This function accepts a pfam database file and parses it into a dictionary. Each key is a
    clan and its value is a list of families in that clan. The dict is saved as a pickle file.

    :param pfam: Pfam database file
    ============================================================================================="""

    # Read clans db with csv reader
    clans = {}
    with open(pfam, 'r', encoding='utf8', errors='replace') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            clan = row[1]
            if clan:  # If this family is a part of a clan, add family to clan dict
                if clan in clans:
                    clans[clan].append(row[3])
                else:  # Initialize new key
                    clans[clan] = [row[3]]

    # Save dict with pickle
    with open('data/clans.pkl', 'wb') as file:
        pickle.dump(clans, file)


def main():
    """=============================================================================================
    Main detects if Pfam-A.full database is in directory. If not, it will download from Pfam
    website and unzip. Then, it will call read_pfam to parse each family into individual fasta
    files.
    ============================================================================================="""

    # Read Pfam-A.clans if it exists
    pfam_clans = 'data/Pfam-A.clans.tsv'
    if not os.path.exists('data/Pfam-A.clans.tsv'):
        print('Pfam-A.seed not found. Downloading from Pfam...')
        if not os.path.exists('data'):
            os.mkdir('data')
        os.system('wget -P data ' \
            'https://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam35.0/Pfam-A.clans.tsv.gz')
        os.system('gunzip data/Pfam-A.clans.tsv.gz')

    # Parse clans and save them as dict
    read_pfam(pfam_clans)


if __name__ == '__main__':
    main()
