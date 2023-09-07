"""Transforms a directory of embeddings with DCT and saves the transformations.

__author__ = "Ben Iovino"
__date__ = "09/07/23"
"""

import os
import numpy as np
from util import Transform


def transform_embed(edirec: str, s1: int, s2: int):
    """Saves DCT transformations of embeddings.
    """

    # Directory for transforms
    dir_info = f"{'_'.join(edirec.split('_')[:-1])}_transform"
    os.mkdir(dir_info)

    # Get embeds for each fam and transform each one individually
    for fam in os.listdir(edirec):
        embeds = np.load(f'{edirec}/{fam}/embed.npy', allow_pickle=True)
        transforms = []
        for embed in embeds:
            transform = Transform(embed[0], embed[1], None)
            transform.quant_2D(s1, s2)
            transforms.append(transform)

        # Save transforms
        if not os.path.exists(f'{dir_info}/{fam}'):
            os.mkdir(f'{dir_info}/{fam}')
        np.save(f'{dir_info}/{fam}/transform', transforms)


def main():
    """Main
    """


    edirec = 'data/esm2_17_embed'
    transform_embed(edirec, 8, 75)


if __name__ == '__main__':
    main()
