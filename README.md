This project uses PEbA (Protein Embedding Based Alignments) to align protein sequences from Pfam and determine
if the resulting alignment scores are able to be used for homology detection.

**************************************************************************************************************
# WORKFLOW
**************************************************************************************************************

parse_seed.py downloads the Pfam-A.seed database file from Pfam if it is not found in the directory. It then
parses the file and extracts the accession number and sequence for each protein. It also extracts the consensus
sequence for each family for later use. Each sequence is read twice, once with gaps and once without gaps.

parse_fasta.py parses the Pfam-A.fasta database file. It extracts the sequence id and family name for each
sequence and saves the fasta sequence in the corresponding family directory.

embed_pfam.py uses the ProtT5-XL-U50 encoder to embed each sequence. It locally downloads the tokenizer and
encoder required if they are not found in the directory. Each embedding is saved as a numpy array in a .txt file.

avg_embed.py calculates the average embedding for each family and saves it as a numpy array in a .txt file.
This is performed by reading the consensus sequence for each family and determining which positions from
each sequence should be included in the average.

align_pfam.py uses PEbA to align sequences and reports the alignment score for each pair of sequences.
read_scores.py reads the alignment scores and plots their distribution.

get_anchors.py compares the embeddings for each sequence in a family to the average embedding for that family
by calculating the cosine similarity between each embedding and the average embedding. Regions of high
similarity are determined by finding consecutive amino acids that have a cosine similarity above a threshold.
The highest scoring regions are used as anchors for the sequence, reducing the size of the sequence.

query_anchors.py allows the user to query a sequence against all of the anchors to determine what family
the query sequence is most similar to. A fasta sequence or pre-embedded sequence can be used as input.

search_anchors.py is a script used to test many queries at once against the anchors.