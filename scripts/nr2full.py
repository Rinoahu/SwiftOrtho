#!/usr/bin/env python
# this script is used to recover blast-like hit from the results of nr sequences blast searching
# usage:
# python this.py foo.sc

import sys

try:
    qry = sys.argv[1]
except:
    print('python this.py foo.sc')
    sys.exit()

def blastparse(f):
    hits = []
    for i in f:
        j = i[:-1].split('\t')
        qds, rds = j[:2]
        if hits and hits[0][0] != qds:
            yield hits
            hits = [j]
        else:
            hits.append(j)
    if hits:
        yield hits

f = open(qry, 'r')
for hits in blastparse(f):
    outs = {}
    for j in hits:
        qds, rds = j[:2]
        for qd in qds.split(';;;'):
            for rd in rds.split(';;;'):
                q = qd.split(' ')[0]
                r = rd.split(' ')[0]
                out = [q, r] + j[2:-2] + [qd, rd]
                #print('\t'.join(out))
                out = '\t'.join(out)
                try:
                    outs[q].append(out)
                except:
                    outs[q] = [out]
    for out in outs.values():
        print(out)
f.close()



f = open(qry, 'r')
for i in f:
    j = i[:-1].split('\t')
    qds, rds = j[:2]
    for qd in qds.split(';;;'):
        for rd in rds.split(';;;'):
            q = qd.split(' ')[0]
            r = rd.split(' ')[0]
            out = [q, r] + j[2:-2] + [qd, rd]
            print('\t'.join(out))
f.close()





#for i in f:
#    j = i[:-1].split('\t')
#    qds = j[-1]
#    for qd in qds.split(' ;;;'):
#            q = qd.split(' ')[0]
#            out = [q] + j[1:-1] + [qd]
#            print('\t'.join(out))
#f.close()

for i in f:
    j = i[:-1].split('\t')
    qds, rds = j[:2]
    for qd in qds.split(';;;'):
        for rd in rds.split(';;;'):
            q = qd.split(' ')[0]
            r = rd.split(' ')[0]
            out = [q, r] + j[2:-2] + [qd, rd]
            print('\t'.join(out))
f.close()

