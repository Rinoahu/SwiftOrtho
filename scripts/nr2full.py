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

f = open(qry, 'r')

for i in f:
    j = i[:-1].split('\t')
    qds, rds = j[-2:]
    for qd in qs.split(' ;;;')
        for rd in rs.split(' ;;;'):
            q = qd.split(' ')[0]
            r = rd.split(' ')[0]
            out = [q, r] + j[2:-2] + [qd, rd]
            print('\t'.join(out))

f.close()
