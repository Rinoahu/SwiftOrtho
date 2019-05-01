#!usr/bin/env python
import sys
qry = sys.argv[1]
f = open(qry, 'r')
for i in f:
    j = i[:-1].split('\t')
    tpe, qid, sid = j[:3]
    qtx, stx = qid.split('|')[0], sid.split('|')[0]
    if tpe == 'O' and qtx == stx:
        print 'error', i[:-1]
    elif tpe == 'IP' and qtx != stx:
        print 'error', i[:-1]
    else:
        print 'right'

