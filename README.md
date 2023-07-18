**************************************************************************************************************
# Searching for Homologous Protein Sequences Using Protein Language Model Embeddings
**************************************************************************************************************

This project uses protein language models to embed sequences from the Pfam-A.seed database. It then uses
these embeddings and other methods to determine if the embeddings can be used for homology detection.

**************************************************************************************************************
# GETTING THE NECESSARY FILES
**************************************************************************************************************

Three scripts are used to download and parse the necessary files from Pfam:

parse_seed.py downloads the Pfam-A.seed database file from Pfam if it is not found in the directory. It then
parses the file and extracts the accession number and sequence for each protein. It also extracts the consensus
sequence for each family for later use. Each sequence is read twice, once with gaps and once without gaps.

parse_fasta.py parses the Pfam-A.fasta database file. It extracts the sequence id and family name for each
sequence and saves the fasta sequence in the corresponding family directory. These sequences are queried
against the database to test homology search.

parse_seed.py parses the Pfam-A.clans.tsv database file. It creates a dictionary mapping each family in a clan
to its clan name. It saves this dictionary as a pickle file. This is used to determine if the results from
a query are in the same clan as the correct family.

**************************************************************************************************************
# EMBEDDING THE SEQUENCES
**************************************************************************************************************

embed_pfam.py uses either ProtT5-XL-U50 or ESM2-t36-3B encoder to embed each sequence from the Pfam-A.seed
database. All embeddings from each family are stored in a single numpy array and saved as a .npy file.

avg_embed.py calculates the average embedding for each family using sequences from the Pfam-A.seed database
and saves it as a numpy array in a .npy file. This is performed by reading the consensus sequence for each
family and determining which positions from each sequence should be included in the average. These positions
from each sequence in the family are then averaged to create the family embedding.

get_anchors.py compares the embeddings for each Pfam-A.seed sequence in a family to the average embedding
for that family by calculating the cosine similarity between each embedding and the average embedding. Regions
of high similarity are determined by finding consecutive amino acids that have a cosine similarity above a
threshold. The highest scoring regions are used as anchors for the average embedding, greatly reducing its size.

dct_avg.py uses the inverse discrete cosine transform to compress the average embeddings to a 1D array.

**************************************************************************************************************
# SEARCHING FOR HOMOLOGOUS SEQUENCES
**************************************************************************************************************

search_anchors.py is used to test many queries (1 random sequence from each family in Pfam.fasta) at once
against the anchors. It reports the total number of searches performed, the number that found a match, the
number that found a match in the top N results, and the number where the first result was not the correct
family but found in the same clan as the correct family.

search_dct.py is also used to test many queries at once against the iDCT average embeddings. It reports the
same results as search_anchors.py.

**************************************************************************************************************
# SEARCH RESULTS
**************************************************************************************************************

So far, searching query sequences against the anchors has not been as successful as desired. To maintain 90%
accuracy (family of the query sequence is in top 10 results), searches take a little over 30 seconds per query.
This is much too slow for practical homolog search, nor is the accuracy high enough. Different filtering
methods have been tried to reduce this time, however, accuracy is reduced as well. The fastest time per query
in our testing has been 20 seconds with 85% accuracy. This is still too slow and inaccurate.

If we cannot reduce the number of anchors searched to reduce search time, then we must find a way to reduce
the size of sequence representations. Using the inverse discrete cosine transform, we can reduce the size of
the embeddings from a large N x 1024 matrix to a single array of with size of our choosing. Just by using the
embeddings produced by the final layer of either ProtT5 or ESM2 and transforming the average embedding for
each family, we can reduce search time to less than 0.1 seconds per query with an accuracy of over 85%. To
improve accuracy, we can concatenate the transformed average embeddings from multiple layers of the encoder
to produce a more informative representation of the family.