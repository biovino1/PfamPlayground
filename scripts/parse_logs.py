"""================================================================================================
This script parses various log files from data directory to get info or graph results.

Ben Iovino   07/25/23   SearchEmb
================================================================================================"""

import argparse
import datetime
import matplotlib.pyplot as plt


def parse_search(file: str) -> dict:  #\\NOSONAR
    """=============================================================================================
    This function gets results from search logs.

    :param file: path to search log file
    :return results: dictionary of results for each query
    ============================================================================================="""

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


def parse_test_layers(file: str):
    """=============================================================================================
    This function gets results from testing.log after running test_layers().

    :param file: path to test_layers.log
    ============================================================================================="""

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
    """=============================================================================================
    This function takes two lists of labels and a list of corresponding values and graphs a heat
    map.

    :param label1: list of labels
    :param label2: list of labels
    :param results: list of values
    :param layer: layer of ESM2
    ============================================================================================="""

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
    """=============================================================================================
    This function gets results from testing.log after running test_transforms().

    :param file: path to test_transforms.log
    ============================================================================================="""

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
    """=============================================================================================
    Main function parses log files from data directory for desired information.
    ============================================================================================="""

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='data/logs/search_dct.log')
    args = parser.parse_args()

    parse_search(args.d)


if __name__ == '__main__':
    main()
