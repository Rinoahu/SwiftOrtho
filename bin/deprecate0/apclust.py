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

from collections import Counter


# print the manual
def manual_print():
    print 'Usage:'
    print '    python mcl.py -i foo.xyz -d 0.5'
    print 'Parameters:'
    print '  -i: tab-delimited file which contain 3 columns'
    print '  -d: inflation parameter.'
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
def apclust(data, KS=-1, damp=.5, itr=100):
    # data:
    # 0: qid, 1: sid, 2: score, 3: R, 4: A

    if KS == -1:
        KS = int(data[:, :2].max()) + 1

    #print 'K is', K, data.shape, data[:, :2].min()
    # 0:row max; 1: index, 2: 2nd row max; 3: index; 4: col sum; 5: diag of r
    diag = np.zeros((KS, 6))
    N, d = data.shape

    #N = 8*10**8
    #itr = 100
    for it in xrange(itr):
        # get row max and 2nd
        for n in xrange(N):
            I, K, s, r, a = data[n]
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
            I, K, s, r, a = data[n]
            i = int(I)
            k = int(K)
            if k != diag[i, 1]:
                r = s - diag[i, 0]
            else:
                r = s - diag[i, 2]

            #tmp = data[n, 3] * (1-damp)
            #data[n, 3] = tmp + damp * r
            data[n, 3] *= (1-damp)
            data[n, 3] += damp * r
            if i == k:
                diag[i, 5] = data[n, 3]

        # get col sum
        for n in xrange(N):
            I, K, s, r, a = data[n]
            if I != K:
                i = int(I)
                k = int(K)
                r = max(0, r)
                diag[k, 4] += r

        # update A
        for n in xrange(N):
            I, K, s, r, a = data[n]
            i = int(I)
            k = int(K)
            #tmp = data[n, 4] * (1-damp)
            data[n, 4] *= (1-damp)
            if i != k:
                #data[n, 4] = tmp + damp * min(0, diag[k, 4] - max(0, data[n, 3]))
                data[n, 4] += damp * min(0, diag[k, 5] + diag[k, 4] - max(0, data[n, 3]))
            else:
                #data[n, 4] = tmp + damp * diag[k, 4]
                data[n, 4] += damp * diag[k, 4]

    RAs = np.zeros(KS)
    clr = np.arange(KS)
    N = 0
    for n in xrange(N):
        I, K, s, r, a = data[n]
        i = int(I)
        k = int(K)
        ra = r + a
        if ra > RAs[i]:
            clr[i] = k
            RAs[i] = ra
            N += 1

    return clr, N

# double and sort the file
# convert fastclust results to matrix
def fc2mat(qry):
    flag = N = 0
    # locus to number
    KK = Counter()
    l2n = {}
    f = open(qry, 'r')
    _o = open(qry + '.npy', 'wb')
    for i in f:
        j = i[:-1].split('\t')
        if len(j) == 4:
            x, y, z = j[1:4]
        else:
            x, y, z = j
        if x > y:
            continue

        if x not in l2n:
            l2n[x] = flag
            flag += 1

        if y not in l2n:
            l2n[y] = flag
            flag += 1

        X, Y = map(l2n.get, [x, y])
        Z = float(z)
        if KK[X] < Z:
            KK[X] = Z

        if KK[Y] < Z:
            KK[Y] = Z

        _o.write(pack('fffff', X, Y, Z, 0, 0))
        _o.write(pack('fffff', Y, X, Z, 0, 0))
        N += 2

    for i, j in KK.items():
        X = Y = i
        Z = j
        _o.write(pack('fffff', X, Y, Z, 0, 0))
        N += 1

    f.close()
    _o.close()

    D = len(l2n)
    return N, D

#os.system('rm %s.ful %s.ful.sort'%(qry, qry))
N, D = fc2mat(qry)
print 'N, D', N, D
N = len(np.memmap(qry+'.npy', mode='r', dtype='float32')) // 5
print 'N is', N
data = np.memmap(qry+'.npy', mode='r+', shape = (N, 5), dtype='float32')
dat = np.asarray(data, dtype = 'float32')
print 'dat0', dat.shape, D
print 'dat1', dat[:, 3:5].sum(1).max()
clr, N = apclust(dat, KS=D)

print 'dat2', np.sum(dat[: 3:5].sum(1)<0)
print len(set(clr)), N

