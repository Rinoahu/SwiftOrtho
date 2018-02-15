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

from array import array

from collections import Counter
import networkx as nx
from cffi import FFI



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

###############################################################################

#sa = np.memmap('test64.npy', shape = len(ta), dtype = 'float32', mode = 'r+b')
#SA = ffi.cast('float32_t *', sa.ctypes.data)

class dmat0:

    def __init__(self, name, shape=None, dtype='i'):
        assert isinstance(shape, tuple)
        n, d = shape
        self.n = n
        self.d = d
        self.dtype = dtype
        #self.data = darray(name, n * d, self.dtype)
        self.ffi = FFI()
        sa = np.memmap(name, shape=n*d, dtype=self.dtype, mode='r+')
        self.data = self.ffi.cast('float *', sa.ctypes.data)
        self.shape = shape
        print 'initial', n * d, len(sa), n, d

    def __getitem__(self, (x, y)):
        n, d = self.n, self.d
        return self.data[x*d+y]

    def __setitem__(self, (x, y), z):
        n, d = self.n, self.d
        self.data[x*d+y] = z

    def shape(self):
        return self.shape



def makemat(name, shape=None, dtype='f'):
    assert isinstance(shape, tuple)
    n, d = shape
    ffi = FFI()
    sa = np.memmap(name, shape=n*d, dtype=dtype, mode='r+')
    data = ffi.cast('float *', sa.ctypes.data)
    return data, n, d



# ap cluster algorithm
@jit
def apclust(data, shape=None, KS=-1, damp=.5, convit=15, itr=100):
    # data:
    # 0: qid, 1: sid, 2: score, 3: R, 4: A
    if KS == -1:
        #KS = int(data[:, :2].max()) + 1
        KS = 10000000

    beta = 1 - damp

    lab = range(KS)
    ras = [float('-inf')] * KS

    # set max convergency iteraion
    mconv = 0
    # print 'K is', K, data.shape, data[:, :2].min()
    # 0:row max; 1: index, 2: 2nd row max; 3: index; 4: col sum; 5: diag of r
    #diag = np.zeros((KS, 6))
    diag = [[0] * 6 for elem in xrange(KS)]
    #N, d = data.shape
    assert shape
    N, d = shape
    dim = d

    #N = 8*10**8
    #itr = 100
    for it in xrange(itr):
        # get row max and 2nd
        for n in xrange(N):
            #I, K, s, r, a = [data[n, elem] for elem in xrange(dim)]
            I, K, s, r, a = [data[n*dim+elem] for elem in xrange(dim)]
            i = int(I)
            k = int(K)
            ra = r + a
            if diag[i][0] < ra:
                diag[i][0] = ra
                diag[i][1] = k
            elif diag[i][2] < ra:
                diag[i][2] = ra
                diag[i][3] = k
            else:
                continue

        # update R
        for n in xrange(N):
            #I, K, s, r, a = [data[n, elem] for elem in xrange(dim)]
            print 'itr', n*dim, N*dim
            I, K, s, r, a = [data[n*dim+elem] for elem in xrange(dim)]
            i = int(I)
            k = int(K)
            if k != diag[i][1]:
                r = s - diag[i][0]
            else:
                r = s - diag[i][2]

            idx = n*dim+3
            data[idx] *= damp
            data[idx] += beta * r
            if i == k:
                #diag[i][5] = data[n, 3]
                diag[i][5] = data[idx]


        for elem in xrange(KS):
            diag[elem][4] = 0

        # get col sum
        for n in xrange(N):
            #I, K, s, r, a = [data[n, elem] for elem in xrange(dim)]
            I, K, s, r, a = [data[n*dim+elem] for elem in xrange(dim)]
            if I != K:
                i = int(I)
                k = int(K)
                r = max(0, r)
                diag[k][4] += r

        # update A
        for n in xrange(N):
            #I, K, s, r, a = data[n, None]
            #I, K, s, r, a = [data[n, elem] for elem in xrange(dim)]
            I, K, s, r, a = [data[n*dim+elem] for elem in xrange(dim)]
            i = int(I)
            k = int(K)
            idx = n*dim+4
            data[idx] *= damp
            if i != k:
                data[idx] += beta * \
                    min(0, diag[k][5] + diag[k][4] - max(0, data[n*dim+3]))
            else:
                data[idx] += beta * diag[k][4]
        # print 'max of RA', np.sum(data[:, 3]>0), np.sum(data[:, 4]>0),
        # (diag[:, 5] > 0).sum()

        # identify exemplar
        #lab = np.arange(KS)
        #ras = np.repeat(-np.inf, KS)
        # for lb in xrange(KS):
        #    lab[lb] = lb
        #ras[:] = -np.inf
        ras = [float('-inf')] * KS
        change = 0
        for n in xrange(N):
            #I, K, s, r, a = data[n, None]
            #I, K, s, r, a = [data[n, elem] for elem in xrange(dim)]
            I, K, s, r, a = [data[n*dim+elem] for elem in xrange(dim)]
            i = int(I)
            k = int(K)
            ra = r + a
            if ras[i] < ra:
                ras[i] = ra
                if lab[i] != k:
                    change = 1
                    lab[i] = k
                else:
                    continue

        # if change == 0:
        #    mconv += 1
        # else:
        #    mconv = 0
        mconv = change == 0 and mconv + 1 or 0
        # print 'mconv', mconv, it
        if mconv > convit:
            break

    return lab

# double and sort the file
# convert fastclust results to matrix


def fc2mat(qry):
    flag = N = 0
    MIN = float('+inf')
    MAX = 0

    # locus to number
    #KK = Counter()
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
        if MIN > Z:
            MIN = Z
        if MAX < X:
            MAX = Z

        # if KK[X] < Z:
        #    KK[X] = Z

        # if KK[Y] < Z:
        #    KK[Y] = Z

        _o.write(pack('fffff', X, Y, Z, 0, 0))
        _o.write(pack('fffff', Y, X, Z, 0, 0))
        N += 2

    #Z = np.median(KK.values())
    # print 'mini relation', MIN, MAX, 2 * MIN - MAX
    #ms = -MIN
    # for i, j in KK.items():
    for i in l2n.values():
        X = Y = i
        #Z = MIN * 2 - MAX
        #Z = MIN
        Z = -10 * 1.5
        _o.write(pack('fffff', X, Y, Z, 0, 0))
        N += 1

    f.close()
    _o.close()

    D = len(l2n)

    n2l = [None] * len(l2n)
    for i, j in l2n.items():
        n2l[j] = i
    return N, D, n2l

#os.system('rm %s.ful %s.ful.sort'%(qry, qry))
N, D, n2l = fc2mat(qry)

#N = len(np.memmap(qry + '.npy', mode='r', dtype='float32')) // 5
#data = np.memmap(qry + '.npy', mode='r+', shape=(N, 5), dtype='float32')
#dat = np.asarray(data, dtype='float32')
dat, n, d = makemat(qry+'.npy', shape=(N, 5), dtype='float')

#print 'dat size', len(dat)

labels = apclust(dat, shape=(n, d), KS=D)

G = nx.Graph()
for i in xrange(len(labels)):
    j = labels[i]
    G.add_edge(i, j)

for i in nx.connected_components(G):
    print '\t'.join([n2l[elem] for elem in i])

