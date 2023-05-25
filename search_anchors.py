"""================================================================================================
This script takes a query sequence and a list of anchor sequences and searches for the most
similar anchor sequence.

Ben Iovino  05/24/23   PfamPlayground
================================================================================================"""

import os
import numpy as np


def query_search(qfamily, query: np.ndarray, anchors: list) -> str:
    """=============================================================================================
    This function takes a query embedding, a directory of anchor embeddings, and finds the most
    similar anchor embedding based on cosine similarity.

    :param query: embedding of query sequence
    :param anchors: list of anchor embeddings
    :return str: name of anchor family with highest similarity
    ============================================================================================="""

    # Search query against every set of anchors
    sims = {}
    for family in os.listdir('Data/anchors'):
        anchors = f'Data/anchors/{family}/anchor_embed.txt'
        anchors_embed = np.loadtxt(anchors)

        # Ignore these for now
        if len(anchors_embed) == 0:
            continue
        if len(anchors_embed) == 1024:
            continue

        if family not in sims:
            sims[family] = []

        # Compare every anchor embedding to query embedding
        max_sim = 0
        for anchor in anchors_embed:
            cos_sim = []
            for embedding in query:
                cos_sim.append(np.dot(anchor, embedding) /
                        (np.linalg.norm(anchor) * np.linalg.norm(embedding)))
            max_sim = max(cos_sim)
            sims[family].append(max_sim)

        # Average similarities
        sims[family] = np.mean(sims[family])

    result = max(sims, key=sims.get)
    if qfamily != result:
        if qfamily in sims:
            print(qfamily, result, sims[qfamily], sims[result])

    # Return key with highest average similarity
    return result


def main():

    correct = 0
    false = 0

    # Load query sequences
    embeddings = 'Data/prott5_embed/'
    for family in os.listdir(embeddings):
        for sequence in os.listdir(f'{embeddings}/{family}'):
            query = np.loadtxt(f'{embeddings}/{family}/{sequence}')
            result = query_search(family, query, 'Data/anchors')
            if result == family:
                correct += 1
            else:
                false += 1

    print(correct, false)


if __name__ == '__main__':
    main()
