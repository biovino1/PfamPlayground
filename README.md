This project uses PEbA (Protein Embedding Based Alignments) to align protein sequences from Pfam and determine
if the resulting alignment scores are able to be used for homology detection.

**************************************************************************************************************
# WORKFLOW
**************************************************************************************************************

parse_pfam.py downloads the Pfam-A.seed database file from Pfam if it is not found in the directory. It then
parses the file and extracts the accession number and sequence for each protein. It also extracts the consensus
sequence for each family for later use.

embed_pfam.py uses the ProtT5-XL-U50 encoder to embed each sequence. It locally downloads the tokenizer and
encoder required if they are not found in the directory. Each embedding is saved as a numpy array in a .txt file.

avg_embed.py calculates the average embedding for each family and saves it as a numpy array in a .txt file.
This is performed by reading the consensus sequence for each family and determining which positions from
each sequence should be included in the average.

align_pfam.py uses PEbA to align sequences and reports the alignment score for each pair of sequences.