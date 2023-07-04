This project uses the ProtT5-XL-U50 encoder to embed sequences from the Pfam-A.seed database. It then uses
these embeddings to create create an averaged embedding for each family, as well as 'anchors' for each
family that represent the most important positions in the family. These anchors are then queried against
using sequences from the Pfam-A.fasta database to determine if the anchors are able to be used for homology
detection.

**************************************************************************************************************
# GETTING THE NECESSARY FILES
**************************************************************************************************************

Three scripts are used to download and parse the necessary files from Pfam:

parse_seed.py downloads the Pfam-A.seed database file from Pfam if it is not found in the directory. It then
parses the file and extracts the accession number and sequence for each protein. It also extracts the consensus
sequence for each family for later use. Each sequence is read twice, once with gaps and once without gaps.
These sequences are later embedded, averaged per family to create a family embedding, and then used to create
anchor positions for the family embedding.

parse_fasta.py parses the Pfam-A.fasta database file. It extracts the sequence id and family name for each
sequence and saves the fasta sequence in the corresponding family directory. These sequences are queried
against the anchors to determine if the anchors are useful for homology detection.

parse_seed.py parses the Pfam-A.clans.tsv database file. It creates a dictionary mapping each family in a clan
to its clan name. It saves this dictionary as a pickle file. This is used to determine if the results from
a query are in the same clan as the correct family.

**************************************************************************************************************
# EMBEDDING AND SEARCHING
**************************************************************************************************************

embed_pfam.py uses the ProtT5-XL-U50 encoder to embed each sequence from the Pfam-A.seed database. It locally
downloads the tokenizer and encoder required if they are not found in the directory. Each embedding is saved
as a numpy array in a binary .npy file.

avg_embed.py calculates the average embedding for each family using sequences from the Pfam-A.seed database
and saves it as a numpy array in a .txt file. This is performed by reading the consensus sequence for each
family and determining which positions from each sequence should be included in the average. These positions
from each sequence in the family are then averaged to create the family embedding.

get_anchors.py compares the embeddings for each Pfam-A.seed sequence in a family to the average embedding
for that family by calculating the cosine similarity between each embedding and the average embedding. Regions
of high similarity are determined by finding consecutive amino acids that have a cosine similarity above a
threshold. The highest scoring regions are used as anchors for the average embedding, greatly reducing its size.

query_anchors.py allows the user to query a sequence against all of the anchors to determine what family
the query sequence is most similar to. A fasta sequence or pre-embedded sequence can be used as input.

search_anchors.py is a script used to test many queries at once against the anchors. It reports the total
number of searches performed, the number that found a match, the number that found a match in the top 10
results, and the number where the first result was not the correct family but found in the same clan
as the correct family.

idct_embed.py takes embeddings and uses the inverse discrete cosine transform to compress the representations.
This is used to reduce the size of the embeddings for filtering purposes during the search.
