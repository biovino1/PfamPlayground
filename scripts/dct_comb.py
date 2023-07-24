"""================================================================================================
This script takes DCT transforms from npy files and combines them into a single numpy array.

Ben Iovino  07/24/23   SearchEmb
================================================================================================"""

import argparse
import numpy as np
from util import Transform


def init_transforms(file: str) -> list:
    """=============================================================================================
    This function takes a file of transforms and loads them as Transform objects.

    :param transforms: list of transform files to load
    :return: dictionary of transforms with sequence as key and list of transforms as value
    ============================================================================================="""

    objects = []
    with open(f'data/{file}', 'rb') as emb:
        transforms = np.load(emb, allow_pickle=True)
        for transform in transforms:
            objects.append(Transform(transform[0], None, transform[1]))

    return objects


def comb_transforms(files: list, objects: list):
    """=============================================================================================
    This function takes a list of transform files and combines them into a single numpy array.

    :param files: list of transform files to load
    :param transforms: list of Transform objects
    ============================================================================================="""

    for file in files:
        with open(f'data/{file}', 'rb') as emb:
            transforms = np.load(emb, allow_pickle=True)

        # Combine transforms with objects
        for i, transform in enumerate(transforms):
            objects[i].concat(transform[1])

    # Save combined transforms
    with open('data/comb_dct.npy', 'wb') as emb:
        np.save(emb, objects)


def main():
    """=============================================================================================
    Main    
    ============================================================================================="""

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=list, default=['avg_dct.npy', 'avg_dct2.npy'])
    args = parser.parse_args()

    objects = init_transforms(args.f[0])
    comb_transforms(args.f[1:], objects)


if __name__ == '__main__':
    main()
