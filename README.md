## Introduction

Fastclust is [orthomcl](http://orthomcl.org/common/downloads/software/v2.0/ "http://orthomcl.org/common/downloads/software/v2.0/")-like tool. It identifies orthologs, paralogs and co-orthologs for genomes.

<!--First, it calls its own fast homologous protein searching tool to do a all-to-all homologous searching. Then, [orthomcl algorithm](https://docs.google.com/document/d/1RB-SqCjBmcpNq-YbOYdFxotHGuU7RK_wqxqDAMjyP_w/pub "https://docs.google.com/document/d/1RB-SqCjBmcpNq-YbOYdFxotHGuU7RK_wqxqDAMjyP_w/pub") is used to identify ortholog, inparalog and co-ortholog protein paris. Finally, [MCL](https://micans.org/mcl "https://micans.org/mcl") is used to group all the protein paris.-->

## Requirement

Make sure that you have the following installed

1. Python2.7 (Recommend [Anaconda](https://www.continuum.io/downloads#linux "https://www.continuum.io/downloads#linux" ) ) and Packages:
    1. [Networkx](https://networkx.github.io/ "https://networkx.github.io/")
    2. [RPython](https://pypi.python.org/pypi/rpython/0.1.4 "https://pypi.python.org/pypi/rpython/0.1.4")
    3. Install packages via pip:

        $ pip install rpython networkx

2. [MCL](https://micans.org/mcl "https://micans.org/mcl")


## Download

    $ git clone https://github.com/Rinoahu/fastclust

## Usage



**1. All-to-all homologous searching:**

        $python fastclust/bin/fast_search.py -i input.fsa -d input.fsa -o input.fsa.sc -e 1e-5 -s 111111

input.fsa is  protein sequences in fasta format. The header of each protein sequence in intput.fsa should in the following format:  
        >xxx|yyyy  
where xxx is the taxon code and yyyy is a sequence identifier. For example:

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


input.fsa.sc is tabular text file. It contains 14 columns. The first 12 columns are the same as blastp -m8 format, the last

2 columns are the lengths of the query and target sequence.  

**2. Find ortholog, inparalog and co-ortholog:**

        $python fastclust/bin/find_orth.py -i input.fsa.sc -c 0.6 -y 0.5 > input.fsa.sc.orth

input.fsa.sc.orth is tabular text file. It contains 4 columns in the following format:

        type    |    gene1    |    gene2    |    score 


type: IP(paralog), O(ortholog) or CO(co-ortholog). 

gene1/gene2: genes' names.

score: normalized bit-score

**3. Cluster:**

        $cut -f2-4 input.fsa.sc.orth > input.fsa.sc.orth.xyz
        $mcl input.fsa.sc.orth.xyz --abc -I 1.5 -o input.fsa.sc.orth.mcl -te 4


input.fsa.sc.orth.mcl is tabular text file. Each row contains protein names. Proteins in the same row belong to the same group.

## Useful tools and pipeline

