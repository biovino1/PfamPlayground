"""================================================================================================
This script takes a query sequence and a list of anchor sequences and searches for the most
similar anchor sequence.

Ben Iovino  05/24/23   PfamPlayground
================================================================================================"""

import os
import numpy as np


def query_search(sequence, qfamily, query: np.ndarray, anchors: list) -> str:
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

    # Get family with max similarity
    result = max(sims, key=sims.get)
    result1_sim = sims[result]
    rcopy = sims.copy()
    rcopy.pop(result)

    # Calculate difference between max and second max
    result2 = max(rcopy, key=sims.get)
    result2_sim = sims[result2]
    difference = result1_sim - result2_sim

    # Print if query family is not result family
    if qfamily != result:
        if qfamily in sims:
            print(f'query: {qfamily}/{sequence}   hit: {result}   sim to real: {round(sims[qfamily], 4)}   sim to result: {round(sims[result], 4)}')

    # Return key with highest average similarity
    return result, difference


def main():

    correct = 0
    false = 0
    tdiff = 0
    runs = 0

    # Load query sequences
    embeddings = 'Data/prott5_embed/'
    for family in os.listdir(embeddings):
        for sequence in os.listdir(f'{embeddings}/{family}'):
            if sequence == 'consensus.txt':
                continue
            query = np.loadtxt(f'{embeddings}/{family}/{sequence}')
            result, diff = query_search(sequence, family, query, 'Data/anchors')
            tdiff += abs(diff)
            if result == family:
                correct += 1
            else:
                false += 1
            runs += 1

    print(f'correct: {correct}   incorrect: {false}   highsim-lowsim: {round(tdiff/runs, 4)}')


if __name__ == '__main__':
    main()
