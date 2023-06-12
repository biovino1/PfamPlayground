"""================================================================================================
This script searches a query sequences from Pfam-A.full against the Pfam-A.seed database using
phmmer. It parses the results and logs the time it took to search each query and the top results.

Ben Iovino  06/12/23   SearchEmb
================================================================================================"""

import datetime
import logging
import os
from random import sample

logging.basicConfig(filename='Data/phmmer_search.log',
                        level=logging.INFO, format='%(message)s')


def parse_results(query):
    """=============================================================================================
    This function reads results from phmmer output and logs the top results and time it took.

    :param query: query sequence
    ============================================================================================="""

    with open('Data/phmmer_results.txt', 'r', encoding='utf8') as file:
        in_results, skip_line = False, False
        logging.info('%s\n%s', datetime.datetime.now(), query)
        for line in file:

            # Read file until you get to top results
            if 'E-value' in line:
                in_results, skip_line = True, True
                continue

            # Skip line after E-value
            if skip_line:
                skip_line = False
                continue

            # Read results
            if in_results:

                # If line has inclusion threshold, stop reading
                if 'inclusion' in line:
                    logging.info('\n')
                    in_results = False
                    break

                # Log result line
                line = line.split()
                logging.info('%s, %s', line[-1], line[0])


def main():

    direc = 'Data/full_seqs'
    for fam in os.listdir(direc):

        # Randomly sample one query from family
        queries = os.listdir(f'{direc}/{fam}')
        query = sample(queries, 1)[0]
        seq_file = f'{direc}/{fam}/{query}'

        # Run phmmer
        os.system(f'phmmer {seq_file} Data/Pfam-A.seed > Data/phmmer_results.txt')
        parse_results(seq_file)

if __name__ == '__main__':
    main()
