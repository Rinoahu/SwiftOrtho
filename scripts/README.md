Some useful scripts:

1. pangenome.py
This is used to identify the core, dispensable and unique genes. It also calculate the statistics of a given pan-genomes.

2. rbh2phy.py
This script uses reciprocal best hit [RBH] to find orthologs by given a taxonomy as reference. Then, it uses famsa/mafft to align all the orthologs. Finally, all the alignments are concatenated to single one.

3. run_all.py
A pipeline which integrate all the scripts mentioned above. 
    1. Call the fast_search.py and find_orth.py to identify all the orthologs, parallogs and co-orthologs.
    2. Call the pangenome.py to identify core, dispensable and specific genes of pan-genome and corresponding statistics like size, openess...
    3. Call the rbh2phy.py to find and align orthologs. Concatenate the alignments to a single one, trim the weak alignment region and call the fasttree to construct the species phylogenetic tree.
    4. If a file that contains operonic information supplied, then it will call the operonclust.py to cluster the operons.
