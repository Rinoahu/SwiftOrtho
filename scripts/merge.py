#!usr/bin/env python
import sys
import os

if len(sys.argv[1:]) < 1:
    print('# this script is used to merge the fasta file into single one.')
    print('# usage:')
    print('  $python this.py dir_name > merged.fsa')
    print('    dir_name is the directory that contains all the fasta files')
    raise SystemExit()


qry = sys.argv[1]

fns = [elem for elem in os.listdir(qry)]
#fns = [elem for elem in os.listdir(qry) if elem.endswith('.fasta')]

flag = 0
for i in fns:
    fn = qry + '/' + i
    if not os.path.isfile(fn):
        continue

    f = open(fn, 'r')
    for j in f:
        if j.startswith('>'):
            print('>' + str(flag) + '|' + j[1:-1])
        else:
            print(j[:-1])

    f.close()
    flag += 1


