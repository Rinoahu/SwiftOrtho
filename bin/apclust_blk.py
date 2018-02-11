#!usr/bin/env python
import os
import sys
from mmap import mmap, ACCESS_WRITE, ACCESS_READ
from struct import pack, unpack
import numpy as np
try:
    from numba import jit
    cc = 'jit'
except:
    jit = lambda x: x
    cc = 'pypy'

from scipy import sparse
#from sklearn.preprocessing import normalize
from collections import Counter
import networkx as nx
from itertools import izip
import gc

try:
    from pysapc import SAP
except:
    SAP = lambda x: x

from cffi import FFI

# mmap based array
ffi = FFI()
ffi.cdef("""
#define PROT_EXEC ...
#define PROT_READ ...
#define PROT_WRITE ...
#define PROT_NONE ...
#define MAP_SHARED ...
#define MAP_PRIVATE ...
#define MAP_ANONYMOUS ...
#define MAP_FIXED ...
#define MAP_GROWSDOWN ...
#define MAP_HUGETLB ...
#define MAP_LOCKED ...
#define MAP_NONBLOCK ...
#define MAP_NORESERVE ...
#define MAP_POPULATE ...
#define MAP_STACK ...
void *mmap(void *addr, size_t length, int prot, int flags, int fd, size_t offset);
void *offset(void *mapped, size_t offset);
""")

C = ffi.verify("""
#include <sys/mman.h>
void *offset(void *mapped, size_t offset) {
    return (void*)((char*)mapped + offset);
}
""")


globals().update({n: getattr(C, n) for n in dir(C)})


def mmap(addr=ffi.NULL, length=0, prot=PROT_NONE, flags=MAP_PRIVATE, fd=0, offset=0, buffer=True):
    m = C.mmap(addr, length, prot, flags, fd, offset)
    if m == -1:
        return None
    if buffer:
        return ffi.buffer(m, length)
    return m





# print the manual
def manual_print():
    print 'Usage:'
    print '    python this.py -i foo.xyz -d 0.5'
    print 'Parameters:'
    print '  -i: tab-delimited file which contain 3 columns'
    print '  -d: damp'
    print '  -p: parameter of preference for apc'
    print '  -I: inflation parameter for mcl'
    print '  -a: algorithm'


argv = sys.argv
# recommand parameter:
args = {'-i': '', '-d': '0.5', '-p': '-10000', '-I': '1.5', '-a': 'alg'}

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
    qry, dmp, prf, ifl, alg = args['-i'], float(args['-d']), float(args['-p']), float(args['-I']), args['-a'].lower()

except:
    manual_print()
    raise SystemExit()


###############################################################################
# pypy version
###############################################################################
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
    L = os.stat(name).st_size
    f = open(name, 'r+')
    m = mmap(prot=PROT_READ|PROT_WRITE, length=L, flags=MAP_SHARED, fd=f.fileno())
    b = ffi.from_buffer(m)
    data = ffi.cast('float *', b)

    return data, n, d


# ap cluster algorithm
# optimized for pypy
@jit
def apclust_pypy(data, shape=None, KS=-1, damp=.5, convit=15, itr=100):
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
        print 'iteraion', it
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
            #print 'itr', n*dim, N*dim
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



###############################################################################
# numba based
###############################################################################

# update RA
# diag[:, 4] = 0
# ras[:] = -np.inf
# chang = 0
@jit
def max_row(data, diag, ras, lab, size=0, damp=.5, beta=.5, mconv=0, change=0):

    #N = data.shape[0]
    N = size
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
@jit
def update_R(data, diag, ras, lab, size=0, damp=.5, beta=.5, mconv=0, change=0):
    N = size
    # update R
    for n in xrange(N):
        I, K, s, r, a = data[n]
        i = int(I)
        k = int(K)
        if k != diag[i, 1]:
            r = s - diag[i, 0]
        else:
            r = s - diag[i, 2]

        data[n, 3] *= damp
        data[n, 3] += beta * r
        if i == k:
            diag[i, 5] = data[n, 3]


# get sum of col
@jit
def sum_col(data, diag, ras, lab, size=0, damp=.5, beta=.5, mconv=0, change=0):
    N = size
    # get col sum
    for n in xrange(N):
        I, K, s, r, a = data[n]
        if I != K:
            i = int(I)
            k = int(K)
            r = max(0, r)
            diag[k, 4] += r

# update A
@jit
def update_A(data, diag, ras, lab, size=0, damp=.5, beta=.5, mconv=0, change=0):
    N = size
    # update A
    for n in xrange(N):
        I, K, s, r, a = data[n]
        i = int(I)
        k = int(K)
        data[n, 4] *= damp
        if i != k:
            data[n, 4] += beta * min(0, diag[k, 5] + diag[k, 4] - max(0, data[n, 3]))
        else:
            data[n, 4] += beta * diag[k, 4]

# changes
@jit
def get_change(data, diag, ras, lab, size=0, damp=.5, beta=.5, mconv=0, change=0):
    N = size
    for n in xrange(N):
        I, K, s, r, a = data[n]
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

    mconv = change == 0 and mconv+1 or 0


# block ap cluster algorithm
@jit
def apclust_blk(dat, KS=-1, damp=.5, convit=15, itr=100, chk=10**8):
    # data:
    # 0: qid, 1: sid, 2: score, 3: R, 4: A
    if KS == -1:
        KS = int(dat[:, :2].max()) + 1

    beta = 1 - damp
    lab = np.arange(KS)
    ras = np.repeat(-np.inf, KS)

    # set max convergency iteraion
    mconv = 0
    change = 0
    #print 'K is', K, data.shape, data[:, :2].min()
    # 0:row max; 1: index, 2: 2nd row max; 3: index; 4: col sum; 5: diag of r
    diag = np.zeros((KS, 6))
    N, d = dat.shape
    data = np.empty((chk, 5))
    for it in xrange(itr):
        #print 'iteration', it
        # get max of row
        for x in xrange(0, N, chk):
            y = min(x+chk, N)
            z = y - x
            data[:z, :] = dat[x:y, :]
            max_row(data, diag, ras, lab, z, damp, beta, mconv)

        # update R
        for x in xrange(0, N, chk):
            y = min(x+chk, N)
            z = y - x
            data[:z, :] = dat[x:y, :]
            update_R(data, diag, ras, lab, z, damp, beta, mconv)
            if z == chk:
                dat[x:y, :] = data
            else:
                dat[x:y, :] = data[:z, :]

        diag[:, 4] = 0
        # get sum of col
        for x in xrange(0, N, chk):
            y = min(x+chk, N)
            z = y - x
            data[:z, :] = dat[x:y, :]
            sum_col(data, diag, ras, lab, z, damp, beta, mconv)

        # update A
        for x in xrange(0, N, chk):
            y = min(x+chk, N)
            z = y - x
            data[:z, :] = dat[x:y, :]
            update_A(data, diag, ras, lab, z, damp, beta, mconv)
            if z == chk:
                dat[x:y, :] = data
            else:
                dat[x:y, :] = data[:z, :]

        # get change
        ras[:] = -np.inf
        chang = 0
        for x in xrange(0, N, chk):
            y = min(x+chk, N)
            z = y - x
            data[:z, :] = dat[x:y, :]
            get_change(data, diag, ras, lab, z, damp, beta, mconv, change)

        if mconv > convit:
            break

    return lab


# ap cluster algorithm
@jit
def apclust(data, KS=-1, damp=.5, convit=15, itr=100):
    # data:
    # 0: qid, 1: sid, 2: score, 3: R, 4: A
    if KS == -1:
        KS = int(data[:, :2].max()) + 1

    beta = 1 - damp

    lab = np.arange(KS)
    ras = np.repeat(-np.inf, KS)

    # set max convergency iteraion
    mconv = 0
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

            data[n, 3] *= damp
            data[n, 3] += beta * r
            if i == k:
                diag[i, 5] = data[n, 3]

        # get col sum
        diag[:, 4] = 0
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
            data[n, 4] *= damp
            if i != k:
                data[n, 4] += beta * min(0, diag[k, 5] + diag[k, 4] - max(0, data[n, 3]))
            else:
                data[n, 4] += beta * diag[k, 4]
        #print 'max of RA', np.sum(data[:, 3]>0), np.sum(data[:, 4]>0), (diag[:, 5] > 0).sum()

        # identify exemplar
        #lab = np.arange(KS)
        #ras = np.repeat(-np.inf, KS)
        #for lb in xrange(KS):
        #    lab[lb] = lb
        ras[:] = -np.inf
        change = 0
        for n in xrange(N):
            I, K, s, r, a = data[n]
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

        mconv = change == 0 and mconv+1 or 0
        if mconv > convit:
            break

    return lab


# normalize
def normalize0(x, norm='l1', axis=0):
    ri, ci = x.nonzero()
    cs = x.sum(axis)
    if cs.min() == 0:
        er = cs.nonzero().min()/100.
        cs += er
    if axis == 1:
        cs = cs.T
        idx = ri
    else:
        idx = ci
    y = np.asarray(cs)[0]
    x.data /= y[idx]
        
def normalize(x, norm='l1', axis=0):
    cs = x.sum(axis)
    y = np.asarray(cs)[0]
    if y.min() == 0 and y.max() > 0:
        y += y.nonzero()[0].min()/1e3
    else:
        y += 1e-8

    x.data /= y.take(x.indices, mode='clip')


# mcl cluster based on sparse matrix
#    x: dok matrix
#    I: inflation parameter
#    E: expension parameter
#    P: 1e-5. threshold to prune weak edge
def mcl(x, I=1.5, E=2, P=1e-5, rtol=1e-5, atol=1e-8, itr=100, check=5):

    for i in xrange(itr):

        #print 'iteration', i

        # normalization of col
        normalize(x, norm='l1', axis=0)
        if i % check == 0:
            x_old = x.copy()

        # expension
        x **= E

        # inflation
        #x = x.power(I)
        x.data **= I
        #print 'max cell', x.data.max()

        # stop if no change
        if i % check == 0 and i > 0:
            #print 'iteration', i, x.max(), x.min()
            if (abs(x-x_old)-rtol * abs(x_old)).max() <= atol:
                break
            # prune weak edge
            #x.data[x.data < P] = 0.

        # prune weak edge
        x.data[x.data < P] = 0.


    # get cluster
    G = nx.Graph()
    rows, cols = x.nonzero()
    vals = x.data
    for i, j, k in izip(rows, cols, vals):
        if k > P:
            G.add_edge(i, j)

    return G


# double and sort the file
# convert fastclust results to matrix
def fc2mat0(qry, prefer=-10000):
    flag = N = 0
    MIN = float('+inf')
    MAX = 0

    # locus to number
    #KK = Counter()
    l2n = {}
    txs = set()
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
        qtx = x.split('|')[0]
        stx = y.split('|')[0]
        txs.add(qtx)
        txs.add(stx)
        if x not in l2n:
            l2n[x] = flag
            flag += 1

        if y not in l2n:
            l2n[y] = flag
            flag += 1

        X, Y = map(l2n.get, [x, y])
        Z = float(z)
        #if MIN > Z:
        #    MIN = Z
        #if MAX < X:
        #    MAX = Z

        #if KK[X] < Z:
        #    KK[X] = Z

        #if KK[Y] < Z:
        #    KK[Y] = Z

        _o.write(pack('fffff', X, Y, Z, 0, 0))
        _o.write(pack('fffff', Y, X, Z, 0, 0))
        N += 2

    #Z = prefer
    Z = len(txs) * -10
    #print 'Z is', Z
    #Z = np.median(KK.values())
    #ms = -MIN
    #for i, j in KK.items():
    for i in l2n.values():
        X = Y = i
        #Z = MIN * 2 - MAX
        #Z = MIN
        #Z = -10000
        _o.write(pack('fffff', X, Y, Z, 0, 0))
        N += 1

    f.close()
    _o.close()

    D = len(l2n)
    
    n2l = [None] * len(l2n)
    for i, j in l2n.items():
        n2l[j] = i
    return N, D, n2l


def fc2mat(qry, prefer=-10000, alg='mcl'):
    flag = N = 0
    MIN = float('+inf')
    MAX = 0

    # locus to number
    #KK = Counter()
    # nearest neighbors
    NNs = {}
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
        _o.write(pack('fffff', X, Y, Z, 0, 0))
        _o.write(pack('fffff', Y, X, Z, 0, 0))
        N += 2
        if X in NNs:
            if Z > NNs[X][0]:
                NNs[X] = [Z, Y]
            elif Z == NNs[X][0]:
                NNs[X].append(Y)
            else:
                pass
        else:
            NNs[X] = [Z, Y]

        if Y in NNs:
            if Z > NNs[Y][0]:
                NNs[Y] = [Z, X]
            elif Z == NNs[Y][0]:
                NNs[Y].append(X)
            else:
                pass
        else:
            NNs[Y] = [Z, X]

    # set preference
    G = nx.Graph()
    prfs = [0] * len(l2n)
    for X in NNs:
        j = NNs[X]
        Z = j[0]
        for Y in j[1:]:
            prfs[Y] += Z
            prfs[X] -= Z
            G.add_edge(X, Y)

    Prf = len(set([elem.split('|')[0] for elem in l2n])) * -20.
    if alg == 'apc' or alg == 'sap':
        for i in xrange(len(l2n)):
            X = Y = i
            #Z = prfs[i]
            #Z = -len(prfs)
            Z = Prf
            _o.write(pack('fffff', X, Y, Z, 0, 0))
            N += 1

    f.close()
    _o.close()

    D = len(l2n)
    
    n2l = [None] * len(l2n)
    for i, j in l2n.items():
        n2l[j] = i
    return N, D, n2l

# get connect components from graph for mcl
def cnc0(qry, alg='mcl'):
    flag = N = 0
    # locus to number
    # nearest neighbors
    NNs = {}
    f = open(qry, 'r')
    for i in f:
        j = i[:-1].split('\t')
        if len(j) == 4:
            x, y, z = j[1:4]
        else:
            x, y, z = j[:3]

        if x > y:
            continue

        Z = float(z)
        if x in NNs:
            if Z > NNs[x][0]:
                NNs[x] = [Z, y]
            elif Z == NNs[x][0]:
                NNs[x].append(y)
            else:
                pass
        else:
            NNs[x] = [Z, y]

        if y in NNs:
            if Z > NNs[y][0]:
                NNs[y] = [Z, x]
            elif Z == NNs[y][0]:
                NNs[y].append(x)
            else:
                pass
        else:
            NNs[y] = [Z, x]

    f.close()
    # get 1st round of cluster
    G = nx.Graph()
    for x, j in NNs.iteritems():
        for y in j[1:]:
            G.add_edge(x, y)

    l2n = {}
    cnt = {}
    flag = 0
    for i in nx.connected_components(G):
        cnt[flag] = len(i)
        for j in i:
            l2n[j] = flag
        flag += 1


    # get second round of cluster
    del G
    gc.collect()

    D = flag
    G_w = sparse.lil_matrix((D, D), dtype='float32')
    G_d = sparse.lil_matrix((D, D), dtype='float32')

    f = open(qry, 'r')
    for i in f:
        j = i[:-1].split('\t')
        if len(j) == 4:
            x, y, z = j[1:4]
        else:
            x, y, z = j

        if x > y:
            continue

        X, Y = map(l2n.get, [x, y])
        Z = float(z)
        a, b = cnt[X], cnt[Y]
        c = (a * b) ** .5
        #c = a * b
        G_w[X, Y] += Z / c
        G_w[Y, X] = G_w[X, Y]
        G_d[X, Y] += 1. / c
        G_d[Y, X] = G_d[X, Y]

    f.close()
    # use mcl to cluster degree based similar
    # print G_d.nonzero(), G_d.data
    # print G_w.nonzero(), G_w.data
    G_d = G_d.tocsr()
    G = mcl(G_d, I=1.2)
    n2n = {}
    flag = 0
    for i in nx.connected_components(G):
        for j in i:
            n2n[j] = flag
        flag += 1

    # clusters
    cls = {}
    for i, j in l2n.iteritems():
        try:
            cls[j].append(i)
        except:
            cls[j] = [i]

    cls = {}
    for i in l2n:
        j = l2n[i]
        try:
            #l2n[i] = n2n[j]
            c = n2n[j]
        except:
            #l2n[i] = flag
            c = flag
            flag += 1
        cls[i] = c


    #print 'cluster size', len(cls)
    N = 0
    n2l = []
    for i in l2n.iterkeys():
        l2n[i] = N
        n2l.append(i)
        N += 1

    #D = flag
    D = len(n2l)
    G_d = sparse.lil_matrix((D, D), dtype='float32')
    #print 'shape', G_d.shape

    f = open(qry, 'r')
    for i in f:
        j = i[:-1].split('\t')
        if len(j) == 4:
            x, y, z = j[1:4]
        else:
            x, y, z = j

        if x > y:
            continue

        cx, cy = map(cls.get, [x, y])
        if cx == cy:
            X, Y = map(l2n.get, [x, y])
            Z = float(z)
            G_d[X, Y] = Z
            G_d[Y, X] = Z
            #print 'pair', X, Y, Z

    G_d = G_d.tocsr()
    #print G_d.nonzero(), G_d.data

    G = mcl(G_d, I=1.5)
    #print G.size()
    for i in nx.connected_components(G):
        print '\t'.join([n2l[elem] for elem in i])


    return G_w, G_d


def cnc1(qry, alg='mcl'):
    flag = N = 0
    # locus to number
    # nearest neighbors
    NNs = {}
    f = open(qry, 'r')
    for i in f:
        j = i[:-1].split('\t')
        if len(j) == 4:
            x, y, z = j[1:4]
        else:
            x, y, z = j[:3]

        if x > y:
            continue

        Z = float(z)
        if x in NNs:
            if Z > NNs[x][0]:
                NNs[x] = [Z, y]
            elif Z == NNs[x][0]:
                NNs[x].append(y)
            else:
                pass
        else:
            NNs[x] = [Z, y]

        if y in NNs:
            if Z > NNs[y][0]:
                NNs[y] = [Z, x]
            elif Z == NNs[y][0]:
                NNs[y].append(x)
            else:
                pass
        else:
            NNs[y] = [Z, x]

    f.close()
    # get 1st round of cluster
    G = nx.Graph()
    for x, j in NNs.iteritems():
        for y in j[1:]:
            G.add_edge(x, y)

    l2n = {}
    cnt = {}
    flag = 0
    cls = {}
    for i in nx.connected_components(G):
        cnt[flag] = len(i)
        cls[flag] = list(i)
        for j in i:
            l2n[j] = flag

        flag += 1

    # get second round of cluster
    del G
    gc.collect()

    D = flag
    G_w = sparse.lil_matrix((D, D), dtype='float32')
    G_d = sparse.lil_matrix((D, D), dtype='float32')

    f = open(qry, 'r')
    for i in f:
        j = i[:-1].split('\t')
        if len(j) == 4:
            x, y, z = j[1:4]
        else:
            x, y, z = j

        if x > y:
            continue

        X, Y = map(l2n.get, [x, y])
        Z = float(z)
        a, b = cnt[X], cnt[Y]
        c = (a * b) ** .5
        #c = a * b
        G_w[X, Y] += Z / c
        G_w[Y, X] = G_w[X, Y]
        G_d[X, Y] += 1. / c
        G_d[Y, X] = G_d[X, Y]

    f.close()



    # use mcl to merge small cluster
    G_d = G_d.tocsr()
    G = mcl(G_d, I=1.2)

    for i in nx.connected_components(G):
        xs = list(i)
        x = xs[0]
        for k in xs[1:]:
            v = cls.pop(k, [])
            cls[x].extend(v)


    #print 'cluster size', len(cls)
    N = 0
    n2l = []
    for i in l2n.iterkeys():
        l2n[i] = N
        n2l.append(i)
        N += 1

    #D = flag
    D = len(n2l)
    G_d = sparse.lil_matrix((D, D), dtype='float32')
    #print 'shape', G_d.shape

    f = open(qry, 'r')
    for i in f:
        j = i[:-1].split('\t')
        if len(j) == 4:
            x, y, z = j[1:4]
        else:
            x, y, z = j

        if x > y:
            continue

        cx, cy = map(cls.get, [x, y])
        if cx == cy:
            X, Y = map(l2n.get, [x, y])
            Z = float(z)
            G_d[X, Y] = Z
            G_d[Y, X] = Z
            #print 'pair', X, Y, Z

    G_d = G_d.tocsr()
    #print G_d.nonzero(), G_d.data

    G = mcl(G_d, I=1.5)
    #print G.size()
    for i in nx.connected_components(G):
        print '\t'.join([n2l[elem] for elem in i])


    return G_w, G_d


# connect components based mcl
def cnc2(qry, alg='mcl'):
    flag = N = 0
    # locus to number
    # nearest neighbors
    NNs = {}
    f = open(qry, 'r')
    for i in f:
        j = i[:-1].split('\t')
        if len(j) == 4:
            x, y, z = j[1:4]
        else:
            x, y, z = j[:3]

        if x > y:
            continue

        Z = float(z)
        if x in NNs:
            if Z > NNs[x][0]:
                NNs[x] = [Z, y]
            elif Z == NNs[x][0]:
                NNs[x].append(y)
            else:
                pass
        else:
            NNs[x] = [Z, y]

        if y in NNs:
            if Z > NNs[y][0]:
                NNs[y] = [Z, x]
            elif Z == NNs[y][0]:
                NNs[y].append(x)
            else:
                pass
        else:
            NNs[y] = [Z, x]

    f.close()
    # get 1st round of cluster
    G = nx.Graph()
    for x, j in NNs.iteritems():
        for y in j[1:]:
            G.add_edge(x, y)

    l2n = {}
    cnt = {}
    flag = 0
    cls = {}
    for i in nx.connected_components(G):
        cnt[flag] = len(i)
        cls[flag] = list(i)
        for j in i:
            l2n[j] = flag

        flag += 1

    # get second round of cluster
    del G
    gc.collect()

    _o = open(qry+'.coo', 'w')
    f = open(qry, 'r')
    for i in f:
        j = i[:-1].split('\t')
        if len(j) == 4:
            x, y, z = j[1:4]
        else:
            x, y, z = j

        if x > y:
            continue

        X, Y = map(l2n.get, [x, y])
        Z = float(z)
        a, b = cnt[X], cnt[Y]
        #c = (a * b) ** .5
        #c = a * b
        wt = Z / a
        dg = 1. / a
        _o.write('%d\t%d\t%f\t%f\n'%(X, Y, wt, dg))
        wt = Z / b
        dg = 1. / b
        _o.write('%d\t%d\t%f\t%f\n'%(Y, X, wt, dg))

    _o.close()
    f.close()

    # sort coo file and merge
    os.system('export LC_ALL=C && sort -n --parallel=8 %s.coo -o %s.coo.srt && rm %s.coo'%(qry, qry, qry))
    _o = open(qry+'.coo', 'wb')
    f = open(qry+'.coo.srt', 'r')
    output = []
    count = 0
    Dmx = 0 
    for i in f:
        j = i[:-1].split('\t')
        x, y, w, d = map(float, j[:4])
        #print 'xyzo', x, y, w, d
        Dmx = max(x, y, Dmx)
        if [x, y] != output[:2]:
            if output:
                #_o.write(pack('ffff', X, Y, w, d))
                a, b, c, d = output[:4]
                _o.write(pack('ffff', a, b, c, d))
                count += 1
            #output = [x, y, w, d]
            output = map(float, j[:4])
            #print 'output1', output
        else:
            output[2] += w
            output[3] += d
        #print 'output', output

    if output:
        #_o.write(pack('ffff', X, Y, w, d))
        #(a, b), (c, d) = output.items()[0]
        #print 'xyz', a, b, c, d
        a, b, c, d = output
        _o.write(pack('ffff', a, b, c, d))

        count += 1

    f.close()
    _o.close()

    COO = np.memmap(qry+'.coo', mode='r', shape=(count, 4), dtype='float32')
    #print 'COO shape', COO.shape, COO[:, :2].max()
    data = COO[:, 3]
    row = np.asarray(COO[:, 0], 'int32')
    col = np.asarray(COO[:, 1], 'int32')
    #print 'Dmx', Dmx, row.max(), col.max()
    Dmx += 1
    G_d = sparse.coo_matrix((data, (row, col)), shape=(Dmx, Dmx), dtype='float32')
    print 'G_d shape', G_d.shape
    Aa, Bb = sparse.csgraph.connected_components(G_d, False)
    print 'components', Aa, len(set(Bb))


    # use mcl to merge small cluster
    #G_d = G_d.tocsr()
    #G = mcl(G_d, I=1.2)

    #for i in nx.connected_components(G):
    #    xs = list(i)
    #    x = xs[0]
    #    for k in xs[1:]:
    #        v = cls.pop(k, [])
    #        cls[x].extend(v)
    print 'connect', Bb.shape
    CLS = {}
    while cls:
        key, val = cls.popitem()
        try:
            CLS[Bb[key]].extend(val)
        except:
            CLS[Bb[key]] = val
    cls = CLS
    print 'connect finish', len(cls)


    #print 'cluster size', len(cls)
    N = 0
    n2l = []
    for i in l2n.iterkeys():
        l2n[i] = N
        n2l.append(i)
        N += 1

    #D = flag
    D = len(n2l)
    G_d = sparse.lil_matrix((D, D), dtype='float32')
    #print 'shape', G_d.shape

    f = open(qry, 'r')
    for i in f:
        j = i[:-1].split('\t')
        if len(j) == 4:
            x, y, z = j[1:4]
        else:
            x, y, z = j

        if x > y:
            continue

        cx, cy = map(cls.get, [x, y])
        if cx == cy:
            X, Y = map(l2n.get, [x, y])
            Z = float(z)
            G_d[X, Y] = Z
            G_d[Y, X] = Z
            #print 'pair', X, Y, Z

    G_d = G_d.tocsr()
    #print G_d.nonzero(), G_d.data

    G = mcl(G_d, I=1.5)
    #print G.size()
    for i in nx.connected_components(G):
        print '\t'.join([n2l[elem] for elem in i])


    return G_w, G_d

# get batch
def batch(f):
    flag = None
    output = []
    for i in f:
        j = i[:-1].split('\t')
        if j[0] != flag:
            if output:
                yield output
            output = [j]
            flag = j[0]
        else:
            output.append(j)

    if output:
        yield output


# mcl wrapper for c,x,y,z format
def mcl_xyz(cxyz):
    l2n = {}
    dmx = 0
    for c,x,y,z in cxyz:
        if x not in l2n:
            l2n[x] = dmx
            dmx += 1
        if y not in l2n:
            l2n[y] = dmx
            dmx += 1

    dmx += 1
    G_d = sparse.lil_matrix((dmx, dmx), dtype='float32')
    for c,x,y,z in cxyz:
        if x > y:
            continue
        X, Y = map(l2n.get, [x, y])
        Z = float(z)
        G_d[X, Y] = Z
        G_d[Y, X] = Z

    n2l = {}
    while l2n:
        key, val = l2n.popitem()
        n2l[val] = key

    G_d = G_d.tocsr()
    G = mcl(G_d, I=1.5)
    for i in nx.connected_components(G):
        print '\t'.join([n2l[elem] for elem in i])




# connected based mcl
def cnc(qry, alg='mcl', chk=100000):
    flag = N = 0
    # locus to number
    # nearest neighbors
    NNs = {}
    f = open(qry, 'r')
    for i in f:
        j = i[:-1].split('\t')
        if len(j) == 4:
            x, y, z = j[1:4]
        else:
            x, y, z = j[:3]

        if x > y:
            continue

        Z = float(z)
        if x in NNs:
            if Z > NNs[x][0]:
                NNs[x] = [Z, y]
            elif Z == NNs[x][0]:
                NNs[x].append(y)
            else:
                pass
        else:
            NNs[x] = [Z, y]

        if y in NNs:
            if Z > NNs[y][0]:
                NNs[y] = [Z, x]
            elif Z == NNs[y][0]:
                NNs[y].append(x)
            else:
                pass
        else:
            NNs[y] = [Z, x]

    f.close()
    # get 1st round of cluster
    G = nx.Graph()
    for x, j in NNs.iteritems():
        for y in j[1:]:
            G.add_edge(x, y)

    l2n = {}
    cnt = {}
    flag = 0
    cls = {}
    for i in nx.connected_components(G):
        cnt[flag] = len(i)
        cls[flag] = list(i)
        for j in i:
            l2n[j] = flag

        flag += 1

    # get second round of cluster
    del G
    gc.collect()

    _o = open(qry+'.coo', 'w')
    f = open(qry, 'r')
    for i in f:
        j = i[:-1].split('\t')
        if len(j) == 4:
            x, y, z = j[1:4]
        else:
            x, y, z = j

        if x > y:
            continue

        X, Y = map(l2n.get, [x, y])
        Z = float(z)
        a, b = cnt[X], cnt[Y]
        #c = (a * b) ** .5
        #c = a * b
        wt = Z / a
        dg = 1. / a
        _o.write('%d\t%d\t%f\t%f\n'%(X, Y, wt, dg))
        wt = Z / b
        dg = 1. / b
        _o.write('%d\t%d\t%f\t%f\n'%(Y, X, wt, dg))

    _o.close()
    f.close()

    # sort coo file and merge
    os.system('export LC_ALL=C && sort -n --parallel=8 %s.coo -o %s.coo.srt && rm %s.coo'%(qry, qry, qry))
    _o = open(qry+'.coo', 'wb')
    f = open(qry+'.coo.srt', 'r')
    output = []
    count = 0
    Dmx = 0 
    for i in f:
        j = i[:-1].split('\t')
        x, y, w, d = map(float, j[:4])
        Dmx = max(x, y, Dmx)
        if [x, y] != output[:2]:
            if output:
                a, b, c, d = output[:4]
                _o.write(pack('ffff', a, b, c, d))
                count += 1
            output = map(float, j[:4])
        else:
            output[2] += w
            output[3] += d

    if output:
        a, b, c, d = output
        _o.write(pack('ffff', a, b, c, d))

        count += 1

    f.close()
    _o.close()

    COO = np.memmap(qry+'.coo', mode='r', shape=(count, 4), dtype='float32')
    data = COO[:, 3]
    row = np.asarray(COO[:, 0], 'int32')
    col = np.asarray(COO[:, 1], 'int32')
    Dmx += 1
    G_d = sparse.coo_matrix((data, (row, col)), shape=(Dmx, Dmx), dtype='float32')
    print 'G_d shape', G_d.shape
    Aa, Bb = sparse.csgraph.connected_components(G_d, False)
    print 'components', Aa, len(set(Bb))


    # use mcl to merge small cluster
    print 'connect', Bb.shape
    CLS = {}
    while cls:
        key, val = cls.popitem()
        try:
            CLS[Bb[key]].extend(val)
        except:
            CLS[Bb[key]] = val

    while CLS:
        key, vals = CLS.popitem()
        for val in vals:
            cls[val] = key

    print 'connect finish', len(cls)
    #print cls

    N = 0
    n2l = []
    for i in l2n.iterkeys():
        l2n[i] = N
        n2l.append(i)
        N += 1

    D = len(n2l)
    _o = open(qry+'.abcd', 'wb')
    f = open(qry, 'r')
    for i in f:
        j = i[:-1].split('\t')
        if len(j) == 4:
            x, y, z = j[1:4]
        else:
            x, y, z = j

        if x > y:
            continue

        cx, cy = map(cls.get, [x, y])
        if cx == cy:
            #X, Y = map(l2n.get, [x, y])
            #Z = float(z)
            #G_d[X, Y] = Z
            #G_d[Y, X] = Z
            #out = '\t'.join(map(str, [cx, X, Y, Z]))
            out = '\t'.join(map(str, [cx, x, y, z]))
            _o.write(out+'\n')

    _o.close()

    #os.system('export LC_ALL=C && sort -n --parallel=8 %s.abcd -o %s.abcd.srt'%(qry, qry))
    os.system('export LC_ALL=C && sort -n --parallel=8 %s.abcd -o %s.abcd.srt && mv %s.abcd.srt %s.abcd'%(qry, qry, qry, qry))
    #raise SystemExit()

    cxyzs = []
    f = open(qry+'.abcd', 'r')
    for cxyz in batch(f):
        cxyzs.extend(cxyz)
        if len(cxyzs) < chk:
            continue
       
        mcl_xyz(cxyzs)
        cxyzs = []
        gc.collect()

    if cxyzs:
        mcl_xyz(cxyzs)
        del cxyzs

    return G_w, G_d



# main function
def main(dat, n2l = None, I=1.5, damp=.62, KS=-1, alg='mcl'):

    if alg == 'mcl':
        #a = dat[:, 2].min()
        #b = dat[:, 2].max()
        #c = b - a
        X = sparse.lil_matrix((D, D), dtype='float32')
        for i in dat:
            x, y, z = i[:3]
            x, y = int(x), int(y)
            #X[x, y] = (z - a)/c
            X[x, y] = z

        X = X.tocsr()
        G = mcl(X, I=I)
        if n2l:
            for i in nx.connected_components(G):
                print '\t'.join([n2l[elem] for elem in i])
        else:
            for i in nx.connected_components(G):
                print '\t'.join(map(str, i))


    elif alg.startswith('ap'):
        labels = apclust_blk(dat, KS=KS, damp=damp, chk=10**8*2)
        G = nx.Graph()
        for i in xrange(len(labels)):
            j = labels[i]
            G.add_edge(i, j)

        if n2l:
            for i in nx.connected_components(G):
                print '\t'.join([n2l[elem] for elem in i])
        else:
            for i in nx.connected_components(G):
                print '\t'.join(map(str, i))

    elif alg == 'sap':
        a = dat[:, 2].min()
        b = dat[:, 2].max()
        c = b - a
        X = sparse.lil_matrix((D, D), dtype='float32')
        for i in dat:
            x, y, z = i[:3]
            x, y = int(x), int(y)
            X[x, y] = (z - a)/c

        X = X.tocsr()
        clf = SAP()
        #clf.preference = 'median'
        Prf = len(set([elem.split('|')[0] for elem in n2l])) * -10.
        clf.preference = np.asarray([Prf] * len(n2l))
        Y = clf.fit_predict(X)
        clsr = {}
        for i in xrange(len(Y)):
            j = n2l[i]
            k = Y[i]
            try:
                clsr[k].append(j)
            except:
                clsr[k] = [j]

        for i in clsr.itervalues():
            print '\t'.join(i)


    else:
        pass


#G_w, G_d = cnc(qry, alg=alg)
#for i in nx.connected_components(G):
#    print i

#raise SystemExit()

#os.system('rm %s.ful %s.ful.sort'%(qry, qry))
N, D, n2l = fc2mat(qry, prf, alg=alg)

if cc == 'jit' or alg == 'mcl':
    #N = len(np.memmap(qry+'.npy', mode='r', dtype='float32')) // 5
    data = np.memmap(qry+'.npy', mode='r+', shape=(N, 5), dtype='float32')
    dat = np.asarray(data, dtype='float32')
    main(dat, n2l=n2l, I=ifl, KS=D, damp=dmp, alg=alg)
else: 
    dat, n, d = makemat(qry+'.npy', shape=(N, 5), dtype='float')
    labels = apclust_pypy(dat, shape=(n, d), KS=D)
    G = nx.Graph()
    for i in xrange(len(labels)):
        j = labels[i]
        G.add_edge(i, j)

    for i in nx.connected_components(G):
        print '\t'.join([n2l[elem] for elem in i])




