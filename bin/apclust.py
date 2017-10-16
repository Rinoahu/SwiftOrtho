#!usr/bin/env python
import os
import sys
from mmap import mmap, ACCESS_WRITE, ACCESS_READ
from struct import pack, unpack
import numpy as np
try:
    from numba import jit
except:
    jit = lambda x: x

# print the manual
def manual_print():
    print 'Usage:'
    print '    python mcl.py -i foo.xyz -d 0.5'
    print 'Parameters:'
    print '  -i: tab-delimited file which contain 3 columns'
    print '  -I: inflation parameter.'
    print '  -a: number of threads'
argv = sys.argv
# recommand parameter:
args = {'-i': '', '-d': '0.5', '-a': '1'}

N = len(argv)
for i in xrange(1, N):
    k = argv[i]
    if k in args:
        try:
            v = argv[i + 1]
        except:
            break
        args[k] = v
    elif k[:2] in args and len(k) > 2:
        args[k[:2]] = k[2:]
    else:
        continue

if args['-i'] == '':
    manual_print()
    raise SystemExit()

try:
    qry, dmp, cpu = args['-i'], float(args['-d']), int(args['-a']),

except:
    manual_print()
    raise SystemExit()

# ap cluster algorithm
@jit
def apclust(data, K=-1, damp=.5, itr=100):
    if K == -1:
        K = int(data[:, 2].max())

    # 0:row max; 1: index, 2: 2nd row max; 3: index; 4: col sum; 5: diag of r
    diag = np.zeros(K, 6)
    N, d = data.shape
    for it in xrange(itr):
        # get row max
        for n in xrange(N):
            I, K, s, r, a = data[n, ]
            i = int(I)
            k = int(K)
            ra = r + a
            if diag[i, 0] < ra:
                diag[i, 0] = ra
                diag[i, 1] = k
            elif diag[i, 2] < ra:
                diag[i, 2] = ra
                diag[i, 3] = k
            else:
                continue

        # update R
        for n in xrange(N):
            I, K, s, r, a = data[n,]
            i = int(I)
            k = int(K)
            if  k != diag[i, 1]:
                r = s - diag[i, 0]
            else:
                r = s - diag[i, 2]
            if i == k:
                diag[i, 5] = r

        # get col sum
        for n in xrange(N):
            I, K, s, r, a = data[n,]
            if I != K:
                i = int(I)
                k = int(K)
                r = max(0, r)
                diag[k, 4] += r

        # update A
        for n in xrange(N):
            I, K, s, r, a = data[n,]
            i = int(I)
            k = int(K)
            if i != k:
                data[n, 4] = min(0, diag[k, 4] - max(0, data[n, 3]))
            else:
                data[n, 4] = diag[k, 4]

    return data

# double and sort the file
flag = N = 0


# locus to number
l2n = {}
f = open(qry, 'r')
_o = open(qry + '.npy', 'wb')
for i in f:
    j = i[:-1].split('\t')
    x, y, z = j

    if x not in l2n:
        l2n[x] = flag
        flag += 1

    if y not in l2n:
        l2n[y] = flag
        flag += 1

    X, Y = map(l2n.get, [x, y])
	Z = float(z)
	_o.write(pack('f', X, Y, Z, 0, 0))
	_o.write(pack('f', Y, X, Z, 0, 0))
    N += 1

f.close()

#os.system('rm %s.ful %s.ful.sort'%(qry, qry))
