"""================================================================================================
This script takes each layer from ESM2-t36-3B and embeds sequences from families that were missed
in prior searches and transforms each family's average embedding with DCT dimensions of 5x44.
Sequences from the full pfam db are then searched against these dct representations and the results
are logged.

Ben Iovino  07/20/23   DCTDomain
================================================================================================"""


import logging
import os

log_filename = 'data/logs/test_layers.log'  #pylint: disable=C0103
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
logging.basicConfig(filename=log_filename, filemode='w',
                     level=logging.INFO, format='%(message)s')


def main():
    """=============================================================================================
    Main
    ============================================================================================="""

    # Embed seqs, transform them, and search
    for i in range(1, 36):
        os.system(f'python scripts/embed_pfam.py -d data/ -e esm2 -l {i}')
        os.system('python scripts/dct_avg.py -d data/esm2_embed -s1 5 -s2 44')
        os.system(f'python scripts/search_dct.py -d data/avg_dct.npy -l {i} -s1 5 -s2 44')

        # Read last line of search log and write to test_layers log
        with open('data/logs/search_dct.log', 'r', encoding='utf8') as file:
            lines = file.readlines()
            last_line = lines[-2]
        logging.info('Search results for ESM2 layer %s\n%s\n', i, last_line)


if __name__ == '__main__':
    main()
