#!usr/bin/env python
# this script is used to extract core genes from the results
# usage:
# python this.py foo.pan foo.clsr input.fsa
import sys
from Bio import SeqIO

try:
    pan, clsr, fsa = sys.argv[1: 4]
except:
    print('python this.py foo.pan foo.clsr input.fsa')
    raise SystemExit

# get pangenome results
pan_res = []
f = open(pan,  'r')
for i in f:
    if i.startswith('#'):
        continue
    j = i[:-1].split('\t', 3)
    grp, typ = j[:2]
    pan_res.append([grp, typ])

f.close()

# get the cluster
f = open(clsr,  'r')
core_genes = []
for i, j in zip(pan_res, f):
    grp, typ = i
    if typ.lower() == 'core':
        core_genes.extend(j[:-1].split('\t'))

f.close()
core_genes = set(core_genes)

for i in SeqIO.parse(fsa, 'fasta'):
    if i.id in core_genes:
        print('>' + i.description)
        print(str(i.seq))