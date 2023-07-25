"""================================================================================================
This script parses log files from data directory for desired information.

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


def parse_test_layers(file: str):  #\\NOSONAR
    """=============================================================================================
    This function gets results from test_layers.log.

    :param file: path to test_layers.log
    :return results: dictionary of results for each query
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


def main():
    """=============================================================================================
    Main
    ============================================================================================="""

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='data/logs/test_layers.log')
    args = parser.parse_args()

    parse_test_layers(args.d)


if __name__ == '__main__':
    main()
