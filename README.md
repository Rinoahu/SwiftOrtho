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
    5. Install packages via pip:

        $ pip install rpython networkx scipy numpy

2. [MCL](https://micans.org/mcl "https://micans.org/mcl")(optional)


## Download

    $ git clone https://github.com/Rinoahu/fastclust

## Usage



**1. All-to-all homologous search:**

    $python fastclust/bin/fast_search.py -i input.fsa -d input.fsa -o input.fsa.sc -e 1e-5 -s 111111

-i|-d: protein sequences in fasta format. The identifier of each protein sequence in intput.fsa should be like this: >xxx|yyyy where xxx is the taxon code and yyyy is a sequence identifier. For example:

    >GCF_000005825.2_ASM582v2|BPOF4_RS00005
    MENIHDLWERALAEMEKKVSKPSYETWLKSTKANDIQNDVITITAPNEFARDWLEEHYAG
    LTSDTIEHLTGARLTPRFVIPQNELEDDFLIEPPKKKKPVSDNGSQNNGTKTMLNDKYTF
    DTFVIGSGNRFAHAASLAVAEAPAKAYNPLFIYGGVGLGKTHLMHAIGHYVMDHNPNAKV
    VYLSSEKFTNEFINSIRDNRAVNFRNKYRNVDVLLIDDIQFLAGKEQTQEEFFHTFNALH
    EESKQIVISSDRPPKEIPTLEDRLRSRFEWGLITDITPPDLETRIAILRKKAKAENLDIP
    NEVMLYIANQIDTNIRELEGALIRVVAYSSLINQDMNADLAAEALKDIIPNSKPKVLTIT
    DIQKLVGEFYHVKLEDFKAKKRTKSVAYPRQIAMYLSREMTDASLPKIGSEFGGRDHTTV
    IHAHEKISKMLTTDQELQQKVQEIMEQLRS
    >GCF_000005825.2_ASM582v2|BPOF4_RS00010
    MQFTIQRDRFVHDVQNVAKAVSSRTTIPILTGIKIVADHEGVTLTGSDSDVSIETFIPKE
    DAENEIVTIEQEGSIVLQARFFAEIVKKLPGETIELIVQDQFATTIRSGSSVFNLNGLDP
    EEYPRLPQLEEDLLFRLPQDMLKNMIRQTVFAVSTQETRPVLTGVNLETEEGELICTATD
    SHRLAMRKATIERNDEELTFSNVVIPGKSLNELSKIIDDSNELIDVVVTENQILFKFKNL
    LFFSRLLEGKYPVTKNMIPAQSKTSFTLKTKPFLQTLERALLLSREGKNNVINLKTLDEG
    LIEITSIQPEVGKVTENIQSEQMQGEDMRISFNGKNIIDALKVIDSEEINIVFTGAMSPF
    VIRPTDHDHYLHLFSPVRTY
    >GCF_000005825.2_ASM582v2|BPOF4_RS00015
    MEKLSISTEYITLGQVLKEVGAIDTGGMAKWYLSEYEVYVNGELENRRGKKLFSGDRVKL
    ADETSIEIVHE


-o: output file. A tabular text file which contains 14 columns. The first 12 columns are the same as blastp -m8 format, the last 2 columns are the lengths of the query and target sequences.  For example:

    GCF_000005825.2_ASM582v2|BPOF4_RS00005  GCF_000005825.2_ASM582v2|BPOF4_RS00005  100.00  450     0       0       1       450     1       450     2.88e-261       897     450     450
    GCF_000005825.2_ASM582v2|BPOF4_RS00005  GCF_000006605.1_ASM660v1|JK_RS00005     53.52   340     158     0       111     450     240     579     1.60e-105       380     450     583
    GCF_000005825.2_ASM582v2|BPOF4_RS00005  GCF_000006645.1_ASM664v1|Y_RS21000      39.91   466     280     19      1       450     14      462     2.66e-99        359     450     462
    GCF_000005825.2_ASM582v2|BPOF4_RS00010  GCF_000005825.2_ASM582v2|BPOF4_RS00010  100.00  380     0       0       1       380     1       380     1.20e-212       735     380     380
    GCF_000005825.2_ASM582v2|BPOF4_RS00015  GCF_000005825.2_ASM582v2|BPOF4_RS00015  100.00  71      0       0       1       71      1       71      1.35e-35        142     71      71
-e: expect value.
-s: space seed pattern.


**2. Find ortholog, inparalog and co-ortholog:**

        $python fastclust/bin/find_orth.py -i input.fsa.sc -c 0.6 -y 0.5 > input.fsa.sc.orth

-i: input file. It is output file of step 1.

-c: alignment coverage of pairwise sequences. 

-y: alignment identity of pairwise sequences.

\>: output file. It is a tabular text file that contains 4 columns in the following format:

        OT  GCF_000006625.1_ASM662v1|UU_RS00560 GCF_000006645.1_ASM664v1|Y_RS14400  1.33510402833
        IP  GCF_000006645.1_ASM664v1|Y_RS19920  GCF_000006645.1_ASM664v1|Y_RS20965  1.23374340949
        CO  GCF_000005845.2_ASM584v2|b0112  GCF_000006645.1_ASM664v1|Y_RS13245  1.41459539212

    Column 1: orthology relationship such as OT(ortholog), CO(co-ortholog), IP(in-paralog).
    Column 2-3: query and target genes' names.
    Column 4: weight of orthology relationship.




**3. Cluster:**

Use  built-in tool to cluster orthology relationships into orthology groups:

        $python fastclust/bin/find_cluster.py -i input.fsa.sc.orth -a mcl -I 1.5 > input.fsa.sc.orth.mcl

-i: a file containing orthology relationships infered by step 2. 

-a: algorithm to cluster. [mcl|apc]

-I: inflation parameters for mcl algorithm. 

\>: output file. this file contains severl rows, each row contains gene names that belong to the same orthology group. 

Or use [MCL](https://micans.org/mcl "https://micans.org/mcl") to cluster orthology relationships into orthology groups:

        $cut -f2-4 input.fsa.sc.orth > input.fsa.sc.orth.xyz
        $mcl input.fsa.sc.orth.xyz --abc -I 1.5 -o input.fsa.sc.orth.mcl -te 4


## Useful tools and pipeline

Run_all.py in directory scripts automatically implements the following steps:

1. all-to-all homologous search.
2. orthology inference.
3. cluster orthology relationships into orthology groups.
4. perform a pan-genome analysis. this step estimate features of pan-genome including gene numbers of core|shared|specific, core size of pan-genome, openess...
5. use conseved proteins to construct species phylogenetic tree.
6. perform operonic clustering if the operonic information supplied.[optional]


Requirement:
1. FastTree 2(http://www.microbesonline.org/fasttree/, "http://www.microbesonline.org/fasttree/").
2. One of the following multiple sequence alignment tools:
    1. [FAMSA](https://github.com/refresh-bio/FAMSA, "https://github.com/refresh-bio/FAMSA"), highly recommended.
    2. [MAFFT](https://mafft.cbrc.jp/alignment/software/, "https://mafft.cbrc.jp/alignment/software/")
    3. [MUSCLE](https://www.drive5.com/muscle/, "https://www.drive5.com/muscle/")

Usage:

        $python run_all.py -i input.fsa -a 4

-i: input file. protein sequences in fasta format.
