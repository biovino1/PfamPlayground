"""Searches fasta queries against a db using mmseqs2

__author__ = "Ben Iovino"
__date__ = "08/18/23"
"""

import datetime
import logging
import os
from Bio import SeqIO


log_filename = 'data/logs/mmseqs.log'  #pylint: disable=C0103
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
logging.basicConfig(filename=log_filename, filemode='w',
                     level=logging.INFO, format='%(message)s')


def main():

    # Read queries.txt and get fasta file
    queries = []
    with open('data/queries.txt', 'r', encoding='utf8') as f:
        for line in f:
            queries.append(line.strip())

    hit_count = 0
    for i, query in enumerate(queries):
        fam = query.split('/')[0]
        seq = query.split('/')[1].split()[0]

        with open(f'data/families_nogaps/{fam}/seqs.fa', 'r', encoding='utf8') as f:
            for record in SeqIO.parse(f, 'fasta'):
                if record.id == seq:
                    query = record

                    # Write query to file
                    with open('data/query.fa', 'w', encoding='utf8') as f:
                        SeqIO.write(query, f, 'fasta')

        # Search db with mmseqs
        #os.system('mmseqs easy-search data/query.fa data/Pfam-A_seed_noq.fasta alnRes.m8 tmp/')

        # Read alnRes.m8 and get top hit
        with open('/home/ben/Desktop/alnRes.m8', 'r', encoding='utf8') as f:
            for line in f:
                top_hit = line.split()[1]
                break

            if top_hit.split('/')[0] == fam:
                hit_count += 1
        logging.info(f'{datetime.datetime.now()}\t{fam}/{query.id}\t{top_hit}\t{hit_count}/{i+1}')


if __name__ == '__main__':
    main()

