#!usr/bin/env python
# remove redundant sequences before analysis

from Bio import SeqIO
import sys


seqs_dct = {}
try:
    qry =sys.argv[1]
except:
    qry = sys.stdin

for i in SeqIO.parse(qry, 'fasta'):
    try:
        #seqs_dct[i.seq].append(i.description)
        seqs_dct[i.seq].append(i.id)
    except:
        #seqs_dct[i.seq] = [i.description]
        seqs_dct[i.seq] = [i.id]

#print(len(seqs_dct))
#sys.exit()

for i in seqs_dct:
    print('>' + ';;;'.join(seqs_dct[i]))
    print(i)
