## Introduction

SwiftOrtho is orthology analysis tool which identifies orthologs, paralogs and co-orthologs for genomes. It is a graph-based approach.

<!--First, it calls its own fast homologous protein searching tool to do a all-to-all homologous searching. Then, [orthomcl algorithm](https://docs.google.com/document/d/1RB-SqCjBmcpNq-YbOYdFxotHGuU7RK_wqxqDAMjyP_w/pub "https://docs.google.com/document/d/1RB-SqCjBmcpNq-YbOYdFxotHGuU7RK_wqxqDAMjyP_w/pub") is used to identify ortholog, inparalog and co-ortholog protein paris. Finally, [MCL](https://micans.org/mcl "https://micans.org/mcl") is used to group all the protein paris.-->

## Requirement

Make sure that you have the following installed

1. Python2.7 (Recommend [Anaconda](https://www.continuum.io/downloads#linux "https://www.continuum.io/downloads#linux" ) ) or [PyPy2.7](http://pypy.org/download.html "http://pypy.org/download.html")(v5.10 or greater) and Packages:
    1. [Networkx](https://networkx.github.io/ "https://networkx.github.io/")
    2. [RPython](https://pypi.python.org/pypi/rpython/0.1.4 "https://pypi.python.org/pypi/rpython/0.1.4")
    3. [numpy](http://www.numpy.org/ "http://www.numpy.org/")
    4. [scipy](https://www.scipy.org/ "https://www.scipy.org/")
	5. [Biopython](http://biopython.org/ "http://biopython.org/")
    6. [sklearn](http://scikit-learn.org/stable/ "http://scikit-learn.org/stable/")
    7. Install packages via pip:

        $ pip install -U rpython networkx scipy numpy biopython sklearn

2. [MCL](https://micans.org/mcl "https://micans.org/mcl")(optional)


## Download

    $ git clone https://github.com/Rinoahu/SwiftOrtho

## Usage



**1. All-to-all homologous search:**

    $python SwiftOrtho/bin/find_hit.py -i input.fsa -d input.fsa -o input.fsa.sc -e 1e-5 -s 111111

-i|-d: protein sequences in fasta format. The identifier of each protein sequence in intput.fsa should be like this: >xxx|yyyy where xxx is the taxon code and yyyy is a sequence identifier. For example:

    >A|a1
    MENIHDLWERALAEMEKKVSKPSYETWLKSTKANDIQNDVITITAPNEFARDWLEEHYAG
    LTSDTIEHLTGARLTPRFVIPQNELEDDFLIEPPKKKKPVSDNGSQNNGTKTMLNDKYTF
	...
    >A|b2
    MQFTIQRDRFVHDVQNVAKAVSSRTTIPILTGIKIVADHEGVTLTGSDSDVSIETFIPKE
    ...

-o: output file. A tabular text file which contains 14 columns. The first 12 columns are the same as blastp -m8 format, the last 2 columns are the lengths of the query and target sequences.  For example:

    A|a1	A|a1	100.00	450	0	0	1	450	1	450	2.88e-261	897	450	450
    A|a1	B|b2	53.52	340	158	0	111	450	240	579	1.60e-105	380	450	583
    ...
-e: expect value.

-s: space seed pattern.


**2. Orthology inference:**

        $python SwiftOrtho/bin/find_orth.py -i input.fsa.sc -c 0.5 -y 0 > input.fsa.sc.orth

-i: input file. It is output file of step 1.

-c: threshold of alignment coverage of pairwise sequences. 

-y: threshold of alignment identity of pairwise sequences.

\>: output file. It is a tabular text file that contains 4 columns in the following format:

        OT	A|a1	B|b1	1.33510402833
        IP	A|a1	A|a2	1.23374340949
        CO	A|a2	B|b2	1.41459539212
        ...

    Col 1: orthology relationship, one of OT(ortholog), CO(co-ortholog) and IP(in-paralog).
    Col 2: identifier of gene in species A.
    Col 3: identifier of gene in species B.
    Col 4: weight of orthology relationship.




**3. Cluster orthology relationships into groups:**

SwiftOrtho implements two cluster algorithms including Markov Clustering and Affinity Aropagation in python. Here is an example of how to use it.

        $python SwiftOrtho/bin/find_cluster.py -i input.fsa.sc.orth -a mcl -I 1.5 > input.fsa.sc.orth.mcl

-i: input file. Input file contains all the orthology relationships. For example:

        OT	A|a1	B|b1	1.33510402833
        IP	A|a1	A|a2	1.23374340949
        CO	A|a2	B|b2	1.41459539212
		...

-a: algorithm to cluster. [mcl|apc]

-I: inflation parameter only for mcl. 

\>: output file. This file contains several rows. Each row stands for an orthologous group. In each row, there are gene identifiers of same or different species. For example:

		A|a1	A|a2	B|b1	C|c1	D|d1
		A|a3	B|b3	C|c3
		A|a4	B|b4
		A|a5	A|a6
		... 

If users prefer [MCL](https://micans.org/mcl "https://micans.org/mcl"), they can follow the steps below:


        $cut -f2-4 input.fsa.sc.orth > input.fsa.sc.orth.xyz
        $mcl input.fsa.sc.orth.xyz --abc -I 1.5 -o input.fsa.sc.orth.mcl -te 4


## Useful tools and pipeline

Run_all.py in directory scripts automatically implements the following steps:

1. all-to-all homologous search.
2. orthology inference.
3. cluster orthology relationships into orthologous groups.
4. perform a pan-genome analysis and estimate main features of pan-genome such as gene numbers of core|shared|specific, core size of pan-genome, openess...
5. use conseved proteins to construct species phylogenetic tree.
6. perform operonic clustering if the operonic information supplied.[optional]


Requirement:
1. [FastTree 2](http://www.microbesonline.org/fasttree/, "http://www.microbesonline.org/fasttree/")
2. One of the following multiple sequence alignment tools:
    1. [FAMSA](https://github.com/refresh-bio/FAMSA, "https://github.com/refresh-bio/FAMSA")(highly recommended)
    2. [MAFFT](https://mafft.cbrc.jp/alignment/software/, "https://mafft.cbrc.jp/alignment/software/")
    3. [MUSCLE](https://www.drive5.com/muscle/, "https://www.drive5.com/muscle/")
3. [trimAl](http://trimal.cgenomics.org/, "http://trimal.cgenomics.org/")

Usage:

        $python run_all.py -i test.fsa -p test.fsa.operon -a 4

-i: input file. protein sequences in fasta format. The identifier of each protein sequence in intput.fsa should be like this: >xxx|yyyy where xxx is the taxon code and yyyy is a sequence identifier.

-p: operonic annotation file. The 1st column of this file should be like x0-->x1-->x2-->x3 or x0<--x1<--x2<--x3. x# stand for gene identifiers and <-- or --> stands for strand of gene. For example:

		A|a0-->A|a1	unknown-->COG1607	unknown::unknown-->I::Acyl-CoA hydrolase::Lipid transport and metabolism
		B|b0<--B|b1	COG4644<--COG1961	X::Transposase and inactivated derivatives, TnpA family::Mobilome: prophages, transposons<--L::Site-specific DNA recombinase related to the DNA invertase Pin::Replication, recombination and repair


-a: number of multi-thread.

Results:
Severl files are generated:
1. test.fsa.sc

	result of all-vs-all homologous search. see example above in Usage section.
2. test.fsa.aln.trim

    concatenation of trimmed aligned protein sequences of conserved genes.
3. test.fsa.nwk

    species phylogenetic tree in newick format. the tree is constructed from the aligned protein sequences of the conserved gene.
4. test.fsa.opc

    orthology relationships. see an example above in Usage section.
5. test.fsa.mcl

    orthologous groups. see an example above in Usage section
6. test.fsa.operon.mcl

    grouped operons which reflect conservation of operons across multiple species. This file contains several rows. Each row stands for an conserved operonic group. In each row, there are operonic information from same or different species. For example:

    	A1-->A2-->A3	B1<--B2<--B3	C1<--C2<--C3
    	A4<--A5<--A6	B4<--B5<--B6	C4<--C5<--C6
    	A7-->A8	B7<--B8<--B9
		....


7. test.fsa.pan

    main features of pan-genome. For example:

        # Statistics and profile of pan-genome:
        # The methods can be found in Hu X, et al. Trajectory and genomic determinants of fungal-pathogen speciation and host adaptation.
        #
        # statistic of core, shared and specific genes:
        # Feature       core    shared  specific        taxon
        # Number        27      2117    9766    5
        #
        # ω(core size of pan-genome) and 95% confidence interval:
        # κc    τc      ω
        # 18001.747907101293±97986.86937584748  0.4604747552601067±0.5879003578202601   29.071595667457963±45.51565446328978
        #
        # θ(new gene number for everay new genome sequenced) and 95% confidence interval:
        # κs    τs      tg(θ)
        # 1334.0072284367752±2342.5492209911768 2.2743910535524314±9.701652708550565    1952.605831944348±1311.6323603805986
        #
        # κ(size and openess of pan-genome, open if γ > 0) and 95% confidence interval:
        # κ     γ
        # 2899.5570049130965±179.58438208536737 0.8785342365438822±0.04423040927927408
        #
        # Type and frequency of each gene group in different species:
        ################################################################################
        #family type    GCF_000005825.2_ASM582v2        GCF_000006645.1_ASM664v1        GCF_000006605.1_ASM660v1        GCF_000005845.2_ASM584v2        GC
        F_000006625.1_ASM662v1
        group_000000000 Share   0       1       0       1       0
        group_000000001 Specific        2       0       0       0       0
        group_000000002 Specific        2       0       0       0       0
        group_000000003 Specific        2       0       0       0       0
        group_000000004 Specific        3       0       0       0       0
		...
