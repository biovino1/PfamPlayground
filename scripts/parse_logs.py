"""This script parses various log files from data directory to get info or graph results.

__author__ = "Ben Iovino"
__date__ = "07/25/23"
"""

import argparse
import datetime
import matplotlib.pyplot as plt
from Bio import SeqIO


def parse_search(file: str) -> dict:  #\\NOSONAR
    """Returns dict of results from search.log.

    :param file: path to search log file
    :return: dictionary of results for each query
    """

    # Get avg time and results
    results = {}
    time, time2, query, avg_time, count = '', '', '', 0, 0
    count = 0
    with open(file, 'r', encoding='utf8') as sefile:
        for line in sefile:

            # Skip line inbetween queries
            if line == '\n':
                time2 = time
                time, query = '', ''
                continue

            # Get time
            if time == '':
                time = line.strip('\n')
                count += 1

                # Subtract time from previous query and add diff to avg_time
                if time2 != '':
                    time1 = datetime.datetime.strptime(time2, '%Y-%m-%d %H:%M:%S.%f')
                    time2 = datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S.%f')
                    avg_time += (time2 - time1).total_seconds()
                    count += 1

            # Get query
            elif query == '':
                query = line.strip('\n')
                results[query] = {}

            # Get top result and its similarity
            else:
                line = line.strip('\n').split(',')
                if results[query] == {}:
                    fam, sim = line[0], line[1].strip('[]')
                    if fam == query.split('/', maxsplit=1)[0]:
                        results[query] = [fam, sim, 'correct']
                    else:
                        results[query] = [fam, sim, 'incorrect']

    # Get average time
    print(f'Average time per query: {avg_time/count}')

    return results


def missed_queries(results: dict):
    """Plots information about the queries that were not found in a search.

    :param results: dictionary where query is key and top result is value
    """

    # Get family names of missed queries
    missed = []
    for query, result in results.items():
        if result[2] == 'incorrect':
            missed.append(query.split('/', maxsplit=1)[0])

    # Read number of sequences in each missed family and their average length
    fams = {}
    for fam in missed:
        with open(f'data/families_nogaps/{fam}/seqs.fa', 'r', encoding='utf8') as fafile:
            count, length = 0, 0
            for seq in SeqIO.parse(fafile, 'fasta'):
                count += 1
                length += len(seq.seq)
            fams[fam] = [count, length/count]

    # Write missed families to file
    with open('data/missed_families.txt', 'w', encoding='utf8') as mfile:
        mfile.write('Family,Number of Sequences,Avg Length\n')
        for fam, counts in fams.items():
            mfile.write(f'{fam},{counts[0]},{int(counts[1])}\n')

    # Graph histogram of number of sequences in each family
    plt.figure(figsize=(12, 5))
    plt.hist([fam[0] for fam in fams.values()], bins=30)
    plt.xlabel('Number of Sequences')
    plt.ylabel('Number of Families')
    plt.title('Number of Sequences in Each Missed Family')
    plt.grid(axis='y', linestyle='--')
    plt.grid(axis='x', linestyle='--')
    plt.axvline(x=64, color='r', linestyle='--')
    plt.show()

    # Plot number of sequences in each family vs. avg sequence length
    plt.figure(figsize=(12, 5))
    plt.plot([fam[0] for fam in fams.values()],
              [fam[1] for fam in fams.values()], 'o', markersize=2)
    plt.xlabel('Number of Sequences')
    plt.ylabel('Avg Length of Sequences')
    plt.title('Number of Sequences vs. Avg Length of Sequences in Each Missed Family')
    plt.grid(axis='y', linestyle='--')
    plt.grid(axis='x', linestyle='--')
    plt.axvline(x=64, color='r', linestyle='--')
    plt.axhline(y=170, color='r', linestyle='--')
    plt.show()


def parse_test_layers(file: str):
    """Plots results about the accuracy of each layer in ESM2.

    :param file: path to test_layers.log
    """

    layers, matches, top = [], [], []
    with open(file, 'r', encoding='utf8') as tefile:
        for line in tefile:

            # Get layer and matches/topn results
            if line.startswith('Search'):
                layers.append(line.strip('\n')[-2:])
            if line.startswith('Queries:'):
                line = line.split()
                query = int(line[1].strip(','))
                matches.append(int(line[3].strip(','))/query)
                top.append(int(line[5].strip(','))/query)

    # Graph results for each layer
    plt.figure(figsize=(12, 5))
    plt.plot(layers, matches,'o-')
    plt.xlabel('ESM2 Layer')
    plt.ylabel('Accuracy')
    plt.grid(axis='y', linestyle='--')
    plt.grid(axis='x', linestyle='--')
    plt.show()


def heat_map(label1: list, label2: list, results: list, layer: int):
    """Plots results from test_transforms() as a heat map.

    :param label1: list of labels
    :param label2: list of labels
    :param results: list of values
    :param layer: layer of ESM2
    """

    # Put values into a 2D array for heat map based on labels
    data = [results[i:i+len(label2)] for i in range(0, len(results), len(label2))]

    # Create heat map from dict
    plt.imshow(data, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.xlabel('col')
    plt.ylabel('rows')
    plt.title(f'Search Results For DCT Dimensions (Layer {layer})')
    plt.xticks(range(len(label2)), label2)
    plt.yticks(range(len(label1)), label1)

    # Place values in each cell
    for i in range(len(label1)):
        for j in range(len(label2)):
            plt.text(j, i, round(data[i][j], 3), ha='center', va='center', color='black')
    plt.show()


def parse_test_transforms(file: str) -> dict:  #\\NOSONAR
    """Returns dict of results from test_transforms.log.

    :param file: path to test_transforms.log
    """

    rows, cols, results, layer = [], [], [], ''
    with open(file, 'r', encoding='utf8') as tefile:
        for line in tefile:
            line = line.split()
            if line == []:  # Ignore empty lines
                continue
            if layer == '':  # Get layer that was transformed
                layer = line[5]
            if line[0] == 'Search':  # Transform information on this line

                # If a different layer, create heat map for previous layer
                if line[5] != layer:
                    heat_map(rows, cols, results, layer)
                    layer = line[5]
                    results = []

                # Get rows, cols, and results
                row, col = line[9].split('x')
                if row not in rows:
                    rows.append(row)
                if col not in cols:
                    cols.append(col)
            if line[0] == 'Queries:':  # Search results on this line
                result = int(line[3].strip(','))/int(line[1].strip(','))
                results.append(result)

    heat_map(rows, cols, results, layer)


def main():
    """Main function parses log files from data directory for desired information.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='data/logs/search_dct.log')
    args = parser.parse_args()

    results = parse_search(args.d)
    missed_queries(results)


if __name__ == '__main__':
    main()
