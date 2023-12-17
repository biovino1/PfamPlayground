**************************************************************************************************************
# Searching for Homologous Protein Sequences Using Protein Language Model Embeddings
**************************************************************************************************************

This project is an exploration into using protein language models to embed sequences from the Pfam v35.0 database (https://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam35.0/) and using these embeddings to determine if they can be used for homology detection.

**************************************************************************************************************
# GETTING THE NECESSARY FILES
**************************************************************************************************************

Three scripts are used to download and parse the necessary files from Pfam:

parse_seed.py downloads the Pfam-A.seed database file from Pfam if it is not found in the directory. It then parses the file and extracts the accession number and sequence for each protein. It also extracts the consensus sequence for each family for later use. Each sequence is read twice, once with gaps and once without gaps.

parse_fasta.py parses the Pfam-A.fasta database file (Pfam full). It extracts the sequence id and family name for each sequence and saves the fasta sequence in the corresponding family directory. These sequences are queried against the database to test homology search.

parse_clans.py parses the Pfam-A.clans.tsv database file. It creates a dictionary mapping each family in a clan to its clan name. It saves this dictionary as a pickle file. This is used to determine if the results from a query are in the same clan as the correct family.

parse_logs.py was used for various tasks during development, mostly to go through search logs and look for queries that had incorrect results for later analysis.

**************************************************************************************************************
# EMBEDDING THE SEQUENCES
**************************************************************************************************************

embed_pfam.py uses either ProtT5-XL-U50 or ESM2-t36-3B encoder to embed each sequence from the Pfam-A.seed database. All embeddings from each family are stored in a single numpy array and saved as a .npy file.

avg_embed.py calculates the average embedding for each family using sequences from the Pfam-A.seed database and saves it as a numpy array in a .npy file. This is performed by reading the consensus sequence for each family and determining which positions from each sequence should be included in the average. These positions from each sequence in the family are then averaged to create the family embedding.

get_anchors.py compares the embeddings for each Pfam-A.seed sequence in a family to the average embeddingfor that family by calculating the cosine similarity between each embedding and the average embedding. Regions of high similarity are determined by finding consecutive amino acids that have a cosine similarity above a threshold. The highest scoring regions are used as anchors for the average embedding, greatly reducing its size.

avg_dct.py uses the inverse discrete cosine transform to compress the average embeddings to a 1D array.

**************************************************************************************************************
# SEARCHING FOR HOMOLOGOUS SEQUENCES
**************************************************************************************************************

search.py is used to test many queries (1 random sequence from each family in Pfam.fasta) at once against a database of DCT's representing each family. It reports the total number of searches performed, the number that found a match, the number that found a match in the top N results, and the number where the first result was not the correct family but found in the same clan as the correct family. 

Using the top results from this DCT search, it can then search against a filtered set of anchor positions from the original embeddings.

**************************************************************************************************************
# SEARCH RESULTS - Anchors
**************************************************************************************************************

Initial searching queried sequences against the anchors was not as successful as desired. To maintain 80% accuracy (family of the query sequence is the top result, also Top1 accuracy), searches take a little over 30 seconds per query. This is much too slow for practical homolog search, nor is the accuracy nearly high enough. Different filtering methods have been tried to reduce this time, the most successful producing an average of 20 seconds per query with a slight decrease in accuracy.

**************************************************************************************************************
# SEARCH RESULTS - iDCT Quantization
**************************************************************************************************************

Inspired by https://github.com/MesihK/prost/, to can use the discrete cosine transform (DCT) to reduce the size of the raw embeddings from a large N x 1024 matrix to a single array of with size of our choosing. Just by using the embeddings produced by the final layer of either ProtT5 or ESM2 and transforming the average embedding for each family, we can reduce search time to less than 0.1 seconds per query with an accuracy of about 78%, almost on par with anchor searching and over 100x faster. To improve accuracy, we can try concatenating the transformed average embeddings from multiple layers of the encoder to produce a larger and hopefully more informative representation of the family.

Embeddings from layer 17 of ESM2-t36-3B with DCT dimensions of 8x75 were found to produce the most accurate search results yet, around 80%. We have tried concatenating embeddings from multiple layers, as each layer captures different information about the sequence, however this led to a decrease in accuracy relative to using only layer 17.

It has been noticed that most of the families missed during searches have a below average number of sequences in the Pfam-A.seed database, or have on average shorter sequences. This could lead to DCT representations that are not as informative as those for families with more/longer sequences. 

To remedy this we can take a higher sample of sequences from the Pfam-A.full database. Although these sequences are not aligned, we can embed each sequence, find it's DCT, and then average all DCTs in a family, as opposed to averaging the consensus embedding where consensus positions were determined from the alignment. Earlier experiments found that the method of averaging does not significantly change search results when the number of sequences per family remains the same. Searching against these reinforced DCT's gave us an accuracy of around 87%, a decent improvement. However, because we are searching with sequences from the Pfam-A.full database, it is possible that a query sequence has been included in the average DCT for its family and is inflating the accuracy. By using more sequences we may also be introducing more noise into the average DCT, which could be hurting accuracy.

**************************************************************************************************************
# Conclusions and Future Directions
**************************************************************************************************************

We compared commonly missed families during searches and compared their ESM2-predicted structures using FATCAT structural alignment (pipeline in comp_str.py). About half (~1800) of the missing querries had one more more families in the top 5 results that it was more structurally similar to than the family it belonged to. This could mean that the query sequences are placed into the wrong family, or that the embeddings are not capturing the correct information. The other half of the missing queries had structure comparisons as expected, where the query was most structurally similar to the family it belonged to, although a third of these had families in the top five that had other families very close to the query structure.

Overall, the way that Pfam was used in this project showed that homology detection with the iDCT quantization method is possible, but faces some difficulties. Other methods generally try to cluster Pfam sequences or remove families that have less than a certain threshold of sequences before performing homology detection tasks, which could have been useful here.

There is also the issue of searching multi-domain proteins, which is a very common task and other embedding based methods have struggled here. Other published methods have used the full sequence for which a Pfam domain was found and search against the database. Using https://github.com/mgtools/DCTdomain, we can predict each domain in a sequence and then find it's DCT and search each one against a database. Further work will use this method on full protein sequences against a database and compare other methods (BLAST, HMMER, etc.) to see how well it performs.

Other methods also have a metric to determine how likely a result is to be homologous. We have been using the Manhattan distance to compare DCT's. We noticed that the distribution of distances between a query sequence and the average DCT for its family is often times much different than the distribution of distances between a query sequence and all other DCT's. This can help us decide if a search result is a true positive or not.
