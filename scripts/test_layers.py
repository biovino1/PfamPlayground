"""================================================================================================
This script takes each layer from ESM2-t36-3B and embeds sequences from families that were missed
in prior searches and transforms each family's average embedding with DCT dimensions of 5x44.
Sequences from the full pfam db are then searched against these dct representations and the results
are logged.

Ben Iovino  07/20/23   DCTDomain
================================================================================================"""


import logging

logging.basicConfig(filename='data/test_layers.log',
                     level=logging.INFO, format='%(message)s')


def main():
    """=============================================================================================
    Main
    ============================================================================================="""

    # Embed, transform, and compute distances
    for i in range(1, 36):
        print(i)


if __name__ == '__main__':
    main()
