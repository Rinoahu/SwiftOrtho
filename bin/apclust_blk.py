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

from scipy import sparse
#from sklearn.preprocessing import normalize
from collections import Counter
import networkx as nx
from itertools import izip

from pysapc import SAP

# print the manual
def manual_print():
    print 'Usage:'
    print '    python mcl.py -i foo.xyz -d 0.5'
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
    if y.min() == 0:
        y += y.nonzero().min()/1e-3

    x.data /= y.take(x.indices, mode='clip')
    #x.data /= y[x.indice]



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

    Prf = len(set([elem.split('|')[0] for elem in l2n])) * -10.
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


    elif alg == 'apc':
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


#os.system('rm %s.ful %s.ful.sort'%(qry, qry))
N, D, n2l = fc2mat(qry, prf, alg=alg)
#D = 6000000

N = len(np.memmap(qry+'.npy', mode='r', dtype='float32')) // 5
data = np.memmap(qry+'.npy', mode='r+', shape=(N, 5), dtype='float32')
dat = np.asarray(data, dtype='float32')


main(dat, n2l=n2l, I=ifl, KS=D, damp=dmp, alg=alg)


raise SystemExit()
###############################################################################
