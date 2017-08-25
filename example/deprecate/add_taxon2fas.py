#!usr/bin/env python
# add species name to m8
import sys
from Bio import SeqIO

if len(sys.argv[1:]) < 1:
    print 'python this.py foo.fsa'
    raise SystemExit()

fasta = sys.argv[1]
f = open(fasta, 'r')
for i in f:
    if i.startswith('>'):
        j = i[i.find('../')+3: i.find('|')]
        hd = i.split(' ')[0][1:]
        print '>'+j+'|'+hd
    else:
        print i[:-1]
f.close()

