#!usr/bin/env python
import sys

qry, ref = sys.argv[1:3]

qN = 0
qry_set = set()
f = open(qry, 'r')
for i in f:
    if '\t' in i:
        j = i[:-1].strip().split('\t')
    else:
        j = i[:-1].strip().split(' ')

    j.sort()
    qry_set.add(tuple(j))
    qN += 1

f.close()

rN = 0
sN = 0
f = open(ref, 'r')
for i in f:

    if '\t' in i:
        j = i[:-1].strip().split('\t')
    else:
        j = i[:-1].strip().split(' ')


    j.sort()
    #qry_set.add(tuple(j))
    if tuple(j) in qry_set:
        sN += 1
    rN += 1

f.close()


#print qN, sN, rN

print 'qry %f'%(sN*100. / qN) + '%', qry
print 'ref %f'%(sN*100. / rN) + '%', ref

