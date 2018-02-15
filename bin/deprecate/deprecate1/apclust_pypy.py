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


class darray:

    # initial the array
    def __init__(self, name, shape=None, dtype='i', chunk=2 ** 30, offset=0):

        self.type = dtype
        self.stride = len(pack(self.type, 0))
        self.f = open(name, 'r+b')
        self.buffer = mmap(self.f.fileno(), 0, access=ACCESS_WRITE)
        if shape:
            self.N = int(shape)
        else:
            self.N = len(self.buffer) // self.stride

        self.chunk = min(int(chunk), self.N)
        self.count = 0
        self.filename = name
        # the offset of array
        self.offset = offset

        # make an tempfile and use mmap to cache the file
        #self.f = ftemp(self.filename, self.N * self.stride)

    def __getitem__(self, idx):

        assert idx < self.N
        #idx += self.offset
        #assert 0 <= idx < self.N

        st_stride = idx * self.stride
        ed_stride = st_stride + self.stride

        # count the operation on array
        self.count += 1
        if self.count >= self.chunk:
            self.__dealloc__()

        return unpack(self.type, self.buffer[st_stride: ed_stride])[0]

    def __delitem__(self, idx):

        assert idx < self.N
        #idx += self.offset
        #assert 0 <= idx < self.N

        st_stride = idx * self.stride
        self.N -= 1

        # count the operation on array
        self.count += 1
        if self.count >= self.chunk:
            self.__dealloc__()

    def __setitem__(self, idx, val):

        assert idx < self.N
        #idx += self.offset
        #assert 0 <= idx < self.N

        st_stride = idx * self.stride
        ed_stride = st_stride + self.stride
        self.buffer[st_stride: ed_stride] = pack(self.type, val)

        # count the operation on array
        self.count += 1
        if self.count >= self.chunk:
            self.__dealloc__()

    def __getslice__(self, st, ed, step=1):

        # if not st:
        #	st = 0
        ed = min(self.N, ed)

        #st += self.offset
        #ed += self.offset

        st_stride = st * self.stride
        ed_stride = ed * self.stride
        step_stride = step * self.stride

        vals = array(self.type)
        for i in xrange(st_stride, ed_stride, step_stride):
            val = unpack(self.type, self.buffer[i: i + self.stride])[0]
            vals.append(val)

            # count the operation on array
            self.count += 1
            if self.count >= self.chunk:
                self.__dealloc__()

        return vals

    def __setslice__(self, st, ed, vals=[]):

        ed = min(self.N, ed)
        #st += self.offset
        #ed += self.offset

        block = abs(ed - st)
        assert len(vals) == block
        self.buffer[st * self.stride: ed *
                    self.stride] = pack(self.type * block, *vals)

        # count the operation on array
        self.count += block * self.stride
        if self.count >= self.chunk:
            self.__dealloc__()

    # release the memory
    def __dealloc__(self):

        self.buffer.close()
        self.f.close()
        self.f = open(self.filename, 'r+b')
        self.buffer = mmap(self.f.fileno(), 0, access=ACCESS_WRITE)
        self.count = 0

    # del the object
    def __del__(self):
        # self.__dealloc__()
        self.buffer.close()
        self.f.close()
        os.remove(self.filename)
        gc.collect()

    def __len__(self):
        return self.N

    # del the array
    def remove(self):

        self.buffer.close()
        # self.f.truncate(0)
        self.f.close()
        os.remove(self.filename)
        gc.collect()

    # write data to disk
    def flush(self):
        self.buffer.flush()
        self.f.flush()

    # iter all elem
    def __iter__(self, st=0, ed=0xffffffff):

        for i in xrange(st, min(self.N, ed)):
            yield self.__getitem__(i)

            if self.count > self.chunk:
                self.__dealloc__()
                self.count = 0

            self.count += 1

    def __repr__(self):
        # pass
        return self[:]

class dmat:

    def __init__(self, name, shape=None, dtype='i'):
        assert isinstance(shape, tuple)
        n, d = shape
        self.n = n
        self.d = d
        self.dtype = dtype
        self.data = darray(name, n * d, self.dtype)
        self.shape = shape

    def __getitem__(self, (x, y)):
        n, d = self.n, self.d
        if y != None:
            return self.data[x*d+y]
        else:
            return self.data[x*d: x*d+d]

    def __setitem__(self, (x, y), z):
        n, d = self.n, self.d
        if y != None:
            self.data[x*d+y] = z
        else:
            self.data[x*d: x*d+d] = z

    def shape(self):
        return self.shape


# ap cluster algorithm
@jit
def apclust(data, KS=-1, damp=.5, convit=15, itr=100):
    # data:
    # 0: qid, 1: sid, 2: score, 3: R, 4: A
    if KS == -1:
        KS = int(data[:, :2].max()) + 1

    beta = 1 - damp

    lab = range(KS)
    ras = [float('-inf')] * KS

    # set max convergency iteraion
    mconv = 0
    # print 'K is', K, data.shape, data[:, :2].min()
    # 0:row max; 1: index, 2: 2nd row max; 3: index; 4: col sum; 5: diag of r
    #diag = np.zeros((KS, 6))
    diag = [[0] * 6 for elem in xrange(KS)]
    N, d = data.shape

    #N = 8*10**8
    #itr = 100
    for it in xrange(itr):
        # get row max and 2nd
        for n in xrange(N):
            I, K, s, r, a = data[n, None]
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
            I, K, s, r, a = data[n, None]
            i = int(I)
            k = int(K)
            if k != diag[i][1]:
                r = s - diag[i][0]
            else:
                r = s - diag[i][2]

            data[n, 3] *= damp
            data[n, 3] += beta * r
            if i == k:
                diag[i][5] = data[n, 3]

        for elem in xrange(KS):
            diag[elem][4] = 0

        # get col sum
        for n in xrange(N):
            I, K, s, r, a = data[n, None]
            if I != K:
                i = int(I)
                k = int(K)
                r = max(0, r)
                diag[k][4] += r

        # update A
        for n in xrange(N):
            I, K, s, r, a = data[n, None]
            i = int(I)
            k = int(K)
            data[n, 4] *= damp
            if i != k:
                data[n, 4] += beta * \
                    min(0, diag[k][5] + diag[k][4] - max(0, data[n, 3]))
            else:
                data[n, 4] += beta * diag[k][4]
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
            I, K, s, r, a = data[n, None]
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
            '''
            if i != k:
                if ras[i] < ra and ras[i] != k:
                    lab[i] = k
                    ras[i] = ra
                    change = 1
            else:
                if ra > 0:
                    if lab[i] != k:
                        change = 1
                    lab[i] = k
                    ras[i] = ra
            '''

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
dat = dmat(qry+'.npy', shape=(N, 5), dtype='f')


labels = apclust(dat, KS=D)

G = nx.Graph()
for i in xrange(len(labels)):
    j = labels[i]
    G.add_edge(i, j)

for i in nx.connected_components(G):
    print '\t'.join([n2l[elem] for elem in i])

