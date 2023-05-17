import os

def main():

    # Get families from embedding directory
    emb_dir = 'prott5_embed'
    families = [f'{emb_dir}/{file}' for file in os.listdir(emb_dir)]

    # Get sequences from each family and align them to each other
    seqs = []
    for family in families:
        seqs.append([f'{family}/{file}' for file in os.listdir(family)])

    # Align average embeddings to all other sequences
    average_embedding = 'prott5_embed/TIP_N/avg_embed.txt'
    embedding_fasta = 'families_nogaps/TIP_N/consensus.fa'
    for group in seqs:
        family = group[0].split('/')[1]

        # Create directories for PEbA and BLOSUM alignments
        if not os.path.exists(f'alignments/PEbA/{family}'):
            os.makedirs(f'alignments/PEbA/{family}')
        if not os.path.exists(f'alignments/blosum/{family}'):
            os.makedirs(f'alignments/blosum/{family}')

        for seq in group:

            # Get fasta files for sample and sequences
            seq_split = seq.split('/')
            seq_fasta = (f'families_nogaps/{seq_split[1]}/{seq_split[2].replace(".txt", ".fa")}')
            if seq_fasta == (f'families_nogaps/{family}/avg_embed.fa'):
                continue

            # Call PEbA
            os.system(f'time python PEbA/peba.py -f1 {embedding_fasta} -f2 {seq_fasta} -e1 {average_embedding} -e2 {seq} -s alignments/PEbA/{family}/{seq_split[2]}')

            # Call BLOSUM
            #os.system(f'python PEbA/local_MATRIX.py -f1 {embedding_fasta} -f2 {seq_fasta} -sf testing/blosum/{family}/{seq_split[2]}')


if __name__ == '__main__':
    main()