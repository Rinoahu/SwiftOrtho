#!usr/bin/env python
import scipy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy import stats
import sys
from time import time
import os
import gc
from struct import pack, unpack
from math import sqrt

from sklearn.externals.joblib import Parallel, delayed
import sharedmem as sm
import multiprocessing as mp
from multiprocessing import Manager, Array

try:
    from numba import jit
except:
    jit = lambda x: x


# parallel matrix A * B
def mul_chk(m):
    x, start, end, y, shape = m
    X = sparse.csr_matrix(x, shape=shape)
    Y = sparse.csr_matrix(y, shape=shape)

    st = time()
    # print 'shape is', X[start: end].shape, Y.shape
    Z = X[start: end] * Y

    return Z


def Pmul(X, Y, chk=64, cache=10**8, cpu=6):
    N, d = X.shape
    D, M = Y.shape
    assert d == D

    if cpu == 1:
        Z = X * Y

    elif X.nnz < cache and Y.nnz < cache:
        Z = X * Y

    else:
        xi = sm.empty(X.data.shape, 'float32')
        xi[:] = X.data
        xj = sm.empty(X.indices.shape, 'int32')
        xj[:] = X.indices
        xk = sm.empty(X.indptr.shape, 'int32')
        xk[:] = X.indptr

        del X
        x = (xi, xj, xk)

        yi = sm.empty(Y.data.shape, 'float32')
        yi[:] = Y.data
        yj = sm.empty(Y.indices.shape, 'int32')
        yj[:] = Y.indices
        yk = sm.empty(Y.indptr.shape, 'int32')
        yk[:] = Y.indptr

        del Y
        y = (yi, yj, yk)

        step = N // chk
        Z = []
        loc = []
        for i in xrange(0, N, step):
            st, ed = i, min(i + step, N)
            loc.append([x, st, ed, y, (N, d)])
            if len(loc) >= 8:
                st = time()
                tmp = Parallel(n_jobs=cpu)(delayed(mul_chk)(elem)
                               for elem in loc)
                Z.extend(tmp)
                loc = []
                del tmp
                gc.collect()

        if len(loc) > 0:
            st = time()
            tmp = Parallel(n_jobs=8)(delayed(mul_chk)(elem) for elem in loc)
            Z.extend(tmp)
            del tmp
            gc.collect()

        Z = sparse.vstack(Z)

    return Z


# given a pairwise relationship, this function will convert the qid, sid into numbers
# and split these relationships into small file
def mat_split0(qry, shape=10**7, step=2 * 10**5, tmp_path=None):
    #_os0 = [open('row_%d.bin'%elem, 'wb') for elem in xrange(6*10**6//step)]
    #_os1 = [open('col_%d.bin'%elem, 'wb') for elem in xrange(6*10**6//step)]
    # build the tmp dir
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    os.system('mkdir -p %s' % tmp_path)

    flag = 0
    q2n = {}
    eye = [0] * shape
    f = open(qry, 'r')
    _oxs, _oys = [], []
    for i in f:
        j = i[:-1].split('\t')
        qid, sid, score = j

        z = float(score)
        if qid not in q2n:
            q2n[qid] = flag
            flag += 1
        if sid not in q2n:
            q2n[sid] = flag
            flag += 1

        x, y = map(q2n.get, [qid, sid])

        out = pack('fff', *[x, y, z])

        xi, yi = x // step, y // step
        try:
            _ox = _oxs[xi]
            _ox.write(out)

        except:
            _o = open(tmp_path + '/%d_row.bin' % xi, 'wb')
            n, m = len(_oxs), xi + 1
            if n < m:
                _oxs.extend([None] * (m - n))

            _oxs[xi] = _o
            _ox = _oxs[xi]
            # print 'xi is', _oxs[xi], _o
            _ox.write(out)

        try:
            _oy = _oys[yi]
            _oy.write(out)
        except:
            _o = open(tmp_path + '/%d_col.bin' % yi, 'wb')
            n, m = len(_oys), yi + 1
            if n < m:
                _oys.extend([None] * (m - n))

            _oys[yi] = _o
            _oy = _oys[yi]
            _oy.write(out)

        # sym
        x, y = y, x
        out = pack('fff', *[x, y, z])
        xi, yi = x // step, y // step

        try:
            _ox = _oxs[xi]
            _ox.write(out)

        except:
            _o = open(tmp_path + '/%d_row.bin' % xi, 'wb')
            n, m = len(_oxs), xi + 1
            if n < m:
                _oxs.extend([None] * (m - n))

            _oxs[xi] = _o
            _ox = _oxs[xi]
            # print 'xi is', _oxs[xi], _o
            _ox.write(out)

        try:
            _oy = _oys[yi]
            _oy.write(out)
        except:
            _o = open(tmp_path + '/%d_col.bin' % yi, 'wb')
            n, m = len(_oys), yi + 1
            if n < m:
                _oys.extend([None] * (m - n))

            _oys[yi] = _o
            _oy = _oys[yi]
            _oy.write(out)

        if eye[x] < z:
            eye[x] = z
        if eye[y] < z:
            eye[y] = z

    # set eye of matrix:
    for i in xrange(0, len(q2n), step):
        zs = q2n[i:i + step]
        xyzs = [[x, x, z] for x, z in zip(xrange(i, i + step), zs)]
        xyzs = sum(xyzs, [])
        out = pack('f' * len(xyzs), *xyzs)
        j = i // step
        _ox = _oxs[j]
        _ox.write(out)
        _oy = _oys[j]
        _oy.write(out)

    for _o in _oxs + _oys:
        try:
            _o.close()
        except:
            continue

    return q2n


def mat_split1(qry, shape=10**8, step=2 * 10**5, tmp_path=None):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    os.system('mkdir -p %s' % tmp_path)

    flag = 0
    q2n = {}
    eye = [0] * shape
    f = open(qry, 'r')
    _os = {}
    for i in f:
        j = i[:-1].split('\t')
        qid, sid, score = j

        z = float(score)
        if qid not in q2n:
            q2n[qid] = flag
            flag += 1
        if sid not in q2n:
            q2n[sid] = flag
            flag += 1

        x, y = map(q2n.get, [qid, sid])
        out = pack('fff', *[x, y, z])

        xi, yi = x // step, y // step
        kxy = (xi, yi)

        try:
            _oxy = _os[kxy]

        except:
            _o = open(tmp_path + '/%d_%d.npz' % (xi, yi), 'wb')
            _os[kxy] = _o
            _oxy = _os[kxy]

        _oxy.write(out)

        # sym
        out = pack('fff', *[y, x, z])
        kyx = (yi, xi)
        try:
            _oyx = _os[kyx]
        except:
            _o = open(tmp_path + '/%d_%d.npz' % (yi, xi), 'wb')
            _os[kyx] = _o
            _oyx = _os[kyx]

        _oyx.write(out)

        if eye[x] < z:
            eye[x] = z
        if eye[y] < z:
            eye[y] = z

    # set eye of matrix:
    for i in xrange(0, len(q2n), step):
        zs = eye[i:i + step]
        xyzs = [[x, x, z] for x, z in zip(xrange(i, i + step), zs)]
        xyzs = sum(xyzs, [])
        out = pack('f' * len(xyzs), *xyzs)
        j = i // step
        _o = _os[(j, j)]
        _o.write(out)

    xy = set()
    for k, _o in _os.items():
        _o.close()
        xy = xy.union(k)

    xy = sorted(xy)
    return q2n, xy


def mat_split2(qry, step=16, tmp_path=None):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    os.system('mkdir -p %s' % tmp_path)
    q2n = {}
    qid_set = set()
    f = open(qry, 'r')
    for i in f:
        j = i[:-1].split('\t')
        if len(j) == 3:
            qid, sid, score = j[:3]
        else:
            qid, sid, score = j[1:4]

        if qid not in qid_set:
            # q2n[qid] = None
            qid_set.add(qid)
        if sid not in qid_set:
            # q2n[sid] = None
            qid_set.add(sid)

    f.close()

    qid_set = list(qid_set)
    qid_set.sort()

    # print qid_set[:10]
    # print 'get all gene id'
    shape = len(qid_set)
    block = shape // step

    # eye = range(shape)
    # for i, j in zip(q2n, eye):
    for i in xrange(shape):
        qid = qid_set[i]
        q2n[qid] = i

    del qid_set
    gc.collect()

    eye = [0] * shape
    _os = {}

    f = open(qry, 'r')
    for i in f:
        j = i[:-1].split('\t')
        if len(j) == 3:
            qid, sid, score = j[:3]
        else:
            qid, sid, score = j[1:4]

        z = float(score)
        x, y = map(q2n.get, [qid, sid])
        out = pack('fff', *[x, y, z])
        # xi, yi = x % block, y % block
        xi, yi = x // block, y // block

        try:
            _ox = _os[xi]
        except:
            _o = open(tmp_path + '/%d.npz' % xi, 'wb')
            _os[xi] = _o
            _ox = _os[xi]

        _ox.write(out)

        # sym
        out = pack('fff', *[y, x, z])
        try:
            _oy = _os[yi]
        except:
            _o = open(tmp_path + '/%d.npz' % yi, 'wb')
            _os[yi] = _o
            _oy = _os[yi]

        _oy.write(out)

        if eye[x] < z:
            eye[x] = z
        if eye[y] < z:
            eye[y] = z

    # set eye of matrix:
    for i in xrange(shape):
        z = eye[i]
        out = pack('fff', *[i, i, z])
        j = i // block
        _o = _os[j]
        _o.write(out)

    '''
    # set eye of matrix:
    chk = 100000
    for i in xrange(0, shape, chk):
        zs = eye[i:i+chk]
        xyzs = [[x,x,z] for x, z in zip(xrange(i, i+chk), zs)]
        xyzs = sum(xyzs, [])
        out = pack('f'*len(xyzs), *xyzs)
        j = i // block
        _o = _os[j]
        _o.write(out)
    '''

    # print 'finish', shape
    # return q2n, xy
    return q2n


def mat_split3(qry, step=4, tmp_path=None):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    os.system('mkdir -p %s' % tmp_path)
    q2n = {}
    qid_set = set()
    f = open(qry, 'r')
    for i in f:
        j = i[:-1].split('\t')
        if len(j) == 3:
            qid, sid, score = j[:3]
        else:
            qid, sid, score = j[1:4]

        if qid not in qid_set:
            # q2n[qid] = None
            qid_set.add(qid)
        if sid not in qid_set:
            # q2n[sid] = None
            qid_set.add(sid)

    f.close()

    qid_set = list(qid_set)
    qid_set.sort()

    # print qid_set[:10]
    # print 'get all gene id'
    shape = len(qid_set)
    block = shape // step + 1

    # eye = range(shape)
    # for i, j in zip(q2n, eye):
    for i in xrange(shape):
        qid = qid_set[i]
        q2n[qid] = i

    del qid_set
    gc.collect()

    eye = [0] * shape
    _os = {}

    f = open(qry, 'r')
    for i in f:
        j = i[:-1].split('\t')
        if len(j) == 3:
            qid, sid, score = j[:3]
        else:
            qid, sid, score = j[1:4]

        z = float(score)
        x, y = map(q2n.get, [qid, sid])
        out = pack('fff', *[x, y, z])
        # xi, yi = x % block, y % block
        xi, yi = x // block, y // block

        try:
            _ox = _os[(xi, yi)]
        except:
            _o = open(tmp_path + '/%d_%d.npz' % (xi, yi), 'wb')
            _os[(xi, yi)] = _o
            _ox = _os[(xi, yi)]

        _ox.write(out)

        # sym
        out = pack('fff', *[y, x, z])
        try:
            _oy = _os[(yi, xi)]
        except:
            _o = open(tmp_path + '/%d_%d.npz' % (yi, xi), 'wb')
            _os[(yi, xi)] = _o
            _oy = _os[(yi, xi)]

        _oy.write(out)

        if eye[x] < z:
            eye[x] = z
        if eye[y] < z:
            eye[y] = z

    # set eye of matrix:
    for i in xrange(shape):
        z = eye[i]
        out = pack('fff', *[i, i, z])
        j = i // block
        try:
            _o = _os[(j, j)]
        except:
            _os[(j, j)] = open(tmp_path + '/%d_%d.npz' % (j, j), 'wb')
            _o = _os[(j, j)]

        _o.write(out)

    # close the file
    for _o in _os.values():
        _o.close()

    # print 'finish', shape
    # return q2n, xy
    return q2n


def mat_split4(qry, step=4, chunk=5 * 10**7, tmp_path=None):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    os.system('mkdir -p %s' % tmp_path)
    q2n = {}
    qid_set = set()
    f = open(qry, 'r')
    for i in f:
        j = i[:-1].split('\t')
        if len(j) == 3:
            qid, sid, score = j[:3]
        else:
            qid, sid, score = j[1:4]

        if qid not in qid_set:
            qid_set.add(qid)

        if sid not in qid_set:
            qid_set.add(sid)

    f.close()

    qid_set = list(qid_set)
    qid_set.sort()
    N = len(qid_set)
    shape = (N, N)
    block = min(N // step + 1, chunk)

    for i in xrange(N):
        qid = qid_set[i]
        q2n[qid] = i

    del qid_set
    gc.collect()

    eye = [0] * N
    _os = {}

    f = open(qry, 'r')
    for i in f:
        j = i[:-1].split('\t')
        if len(j) == 3:
            qid, sid, score = j[:3]
        else:
            qid, sid, score = j[1:4]

        z = float(score)
        x, y = map(q2n.get, [qid, sid])
        out = pack('fff', *[x, y, z])
        xi, yi = x // block, y // block

        try:
            _ox = _os[(xi, yi)]
        except:
            _o = open(tmp_path + '/%d_%d.npz' % (xi, yi), 'wb')
            _os[(xi, yi)] = _o
            _ox = _os[(xi, yi)]

        _ox.write(out)

        # sym
        out = pack('fff', *[y, x, z])
        try:
            _oy = _os[(yi, xi)]
        except:
            _o = open(tmp_path + '/%d_%d.npz' % (yi, xi), 'wb')
            _os[(yi, xi)] = _o
            _oy = _os[(yi, xi)]

        _oy.write(out)

        if eye[x] < z:
            eye[x] = z
        if eye[y] < z:
            eye[y] = z

    # set eye of matrix:
    for i in xrange(N):
        z = eye[i]
        out = pack('fff', *[i, i, z])
        j = i // block
        try:
            _o = _os[(j, j)]
        except:
            _os[(j, j)] = open(tmp_path + '/%d_%d.npz' % (j, j), 'wb')
            _o = _os[(j, j)]

        _o.write(out)

    # close the file
    for _o in _os.values():
        _o.close()

    # reorder the matrix
    cs = None
    fns = [tmp_path + '/' +
        elem for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    # N = len(q2n)
    # shape = (N, N)
    for fn in fns:
        # print 'loading', fns
        g = load_matrix(fn, shape=shape, csr=False)
        ci = csgraph.connected_components(g)
        if cs == None:
            cs = ci
        else:
            cs = merge_connected(cs, ci)

    idx = cs[1].argsort()
    idx_r = np.empty(N, 'int')
    idx_r[idx] = np.arange(N, dtype='int')
    idx = idx_r
    for i in q2n:
        j = q2n[i]
        q2n[i] = idx[j]

    # write reorder matrix
    eye = [0] * N
    _os = {}

    f = open(qry, 'r')
    for i in f:
        j = i[:-1].split('\t')
        if len(j) == 3:
            qid, sid, score = j[:3]
        else:
            qid, sid, score = j[1:4]

        z = float(score)
        x, y = map(q2n.get, [qid, sid])
        out = pack('fff', *[x, y, z])
        xi, yi = x // block, y // block

        try:
            _ox = _os[(xi, yi)]
        except:
            _o = open(tmp_path + '/%d_%d.npz' % (xi, yi), 'wb')
            _os[(xi, yi)] = _o
            _ox = _os[(xi, yi)]

        _ox.write(out)

        # sym
        out = pack('fff', *[y, x, z])
        try:
            _oy = _os[(yi, xi)]
        except:
            _o = open(tmp_path + '/%d_%d.npz' % (yi, xi), 'wb')
            _os[(yi, xi)] = _o
            _oy = _os[(yi, xi)]

        _oy.write(out)

        if eye[x] < z:
            eye[x] = z
        if eye[y] < z:
            eye[y] = z

    # set eye of matrix:
    for i in xrange(N):
        z = eye[i]
        out = pack('fff', *[i, i, z])
        j = i // block
        try:
            _o = _os[(j, j)]
        except:
            _os[(j, j)] = open(tmp_path + '/%d_%d.npz' % (j, j), 'wb')
            _o = _os[(j, j)]

        _o.write(out)

    # close the file
    for _o in _os.values():
        _o.close()

    return q2n


# reorder the matrix
def mat_reorder(qry, q2n, shape=(10**7, 10**7), csr=False, tmp_path=None, step=4, chunk=5*10**7):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    N = shape[0]
    #tsize = os.path.getsize(qry)
    #tstep = tsize // (chunk*12)
    #tstep = max(tsize // (chunk*12), 1)
    #block = min(chunk, N // step + 1)
    #block = min(N//step+1, N//tstep+1)
    block = N // step + 1

    #print 'reorder block', block


    # reorder the matrix
    cs = None
    fns = [tmp_path + '/' +
        elem for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    # N = len(q2n)
    # shape = (N, N)
    for fn in fns:
        # print 'loading', fns
        g = load_matrix(fn, shape=shape, csr=csr)
        ci = csgraph.connected_components(g)
        if cs == None:
            cs = ci
        else:
            cs = merge_connected(cs, ci)

    idx = cs[1].argsort()
    idx_r = np.empty(N, 'int')
    idx_r[idx] = np.arange(N, dtype='int')
    idx = idx_r
    for i in q2n:
        j = q2n[i]
        q2n[i] = idx[j]

    # write reorder matrix
    eye = [0] * N
    _os = {}

    f = open(qry, 'r')
    for i in f:
        j = i[:-1].split('\t')
        if len(j) == 3:
            qid, sid, score = j[:3]
        else:
            qid, sid, score = j[1:4]

        z = float(score)
        x, y = map(q2n.get, [qid, sid])
        out = pack('fff', *[x, y, z])
        xi, yi = x // block, y // block

        try:
            _ox = _os[(xi, yi)]
        except:
            _o = open(tmp_path + '/%d_%d.npz' % (xi, yi), 'wb')
            _os[(xi, yi)] = _o
            _ox = _os[(xi, yi)]

        _ox.write(out)

        # sym
        out = pack('fff', *[y, x, z])
        try:
            _oy = _os[(yi, xi)]
        except:
            _o = open(tmp_path + '/%d_%d.npz' % (yi, xi), 'wb')
            _os[(yi, xi)] = _o
            _oy = _os[(yi, xi)]

        _oy.write(out)

        if eye[x] < z:
            eye[x] = z
        if eye[y] < z:
            eye[y] = z

    # set eye of matrix:
    for i in xrange(N):
        z = eye[i]
        out = pack('fff', *[i, i, z])
        j = i // block
        try:
            _o = _os[(j, j)]
        except:
            _os[(j, j)] = open(tmp_path + '/%d_%d.npz' % (j, j), 'wb')
            _o = _os[(j, j)]

        _o.write(out)

    # close the file
    for _o in _os.values():
        _o.close()

    return q2n


def mat_split(qry, step=4, chunk=5*10**7, tmp_path=None, cpu=4):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    os.system('mkdir -p %s' % tmp_path)
    q2n = {}
    qid_set = set()
    lines = 0
    f = open(qry, 'r')
    for i in f:
        lines += 1
        j = i[:-1].split('\t')
        if len(j) == 3:
            qid, sid, score = j[:3]
        else:
            qid, sid, score = j[1:4]

        if qid not in qid_set:
            qid_set.add(qid)

        if sid not in qid_set:
            qid_set.add(sid)

    f.close()

    qid_set = list(qid_set)
    qid_set.sort()
    N = len(qid_set)
    shape = (N, N)

    # get the size of input file
    #tsize = os.path.getsize(qry)
    #tstep = max(tsize // (chunk*12), 1)
    #block = min(N//step+1, N//tstep+1 )
    if lines*3 > chunk:
        #tstep = max(lines*3//chunk*sqrt(cpu), cpu)
        tstep = sqrt(lines*3//chunk) * sqrt(cpu)
        print 'tstep is', tstep
        tstep = min(max(tstep, 1), 30)
        print 'tstep 2 is', tstep
        #print 'break point', step, tstep, lines * 3, chunk
        #block = min(N//step+1, int(N//tstep)+1)
        block = min(int(N/tstep) + 1, int(N/cpu)+1)
        #block = N // step + 1
        print 'split block cpu=N', block, N // block
    else:
        block = N
        print 'split block cpu=1', block, N // block, lines*3, chunk
        cpu = 1


    for i in xrange(N):
        qid = qid_set[i]
        q2n[qid] = i

    del qid_set
    gc.collect()

    eye = [0] * N
    _os = {}

    f = open(qry, 'r')
    for i in f:
        j = i[:-1].split('\t')
        if len(j) == 3:
            qid, sid, score = j[:3]
        else:
            qid, sid, score = j[1:4]

        z = float(score)
        x, y = map(q2n.get, [qid, sid])
        out = pack('fff', *[x, y, z])
        xi, yi = x // block, y // block

        try:
            _ox = _os[(xi, yi)]
        except:
            _o = open(tmp_path + '/%d_%d.npz' % (xi, yi), 'wb')
            _os[(xi, yi)] = _o
            _ox = _os[(xi, yi)]

        _ox.write(out)

        # sym
        out = pack('fff', *[y, x, z])
        try:
            _oy = _os[(yi, xi)]
        except:
            _o = open(tmp_path + '/%d_%d.npz' % (yi, xi), 'wb')
            _os[(yi, xi)] = _o
            _oy = _os[(yi, xi)]

        _oy.write(out)

        if eye[x] < z:
            eye[x] = z
        if eye[y] < z:
            eye[y] = z

    # set eye of matrix:
    for i in xrange(N):
        z = eye[i]
        out = pack('fff', *[i, i, z])
        j = i // block
        try:
            _o = _os[(j, j)]
        except:
            _os[(j, j)] = open(tmp_path + '/%d_%d.npz' % (j, j), 'wb')
            _o = _os[(j, j)]

        _o.write(out)

    # close the file
    for _o in _os.values():
        _o.close()

    # reorder the matrix
    print 'reorder the matrix'
    #q2n = mat_reorder(qry, q2n, shape, False, tmp_path)

    return q2n


# load sparse matrix from disk
def load_matrix(qry, shape=(10**8, 10**8), csr=False):
    if csr == False:
        n = np.memmap(qry, mode='r+', dtype='float32').shape[0] / 3
        dat = np.memmap(qry, mode='r+', dtype='float32', shape=(n, 3))
        x, y, z = dat[:, 0], dat[:, 1], dat[:, 2]
        x = sparse.csr_matrix((z, (x, y)), shape=shape, dtype='float32')
        # print 'loading shape is', shape, qry, x.shape

    else:
        x = sparse.load_npz(qry)
        # print 'loading shape is', shape, qry, x.shape

    return x


# split row block and col block into row_col block
def preprocess(qry, tmp_path=None):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    fns = os.listdir(tmp_path)
    rows = [elem for elem in fns if elem.endswith('_row.bin')]
    cols = [elem for elem in fns if elem.endswith('_col.bin')]

    for i in rows:
        xn = tmp_path + '/' + i
        x = load_matrix(xn)
        # x += x.transpose()
        for j in cols:
            yn = tmp_path + '/' + j
            y = load_matrix(yn)
            # y += y.transpose()
            z = x * y
            xi = i.split(os.sep)[-1].split('_row')[0]
            yj = j.split(os.sep)[-1].split('_col')[0]
            ij = xi + '_' + yj
            sparse.save_npz(tmp_path + '/' + ij, z)
            del y, z
            gc.collect()

        # remove x
        del x
        gc.collect()
        os.system('rm %s' % xn)

# matrix mul on small blocks


def mul0(qry, shape=(10**7, 10**7), tmp_path=None):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    fns = os.listdir(tmp_path)
    rc = [elem for elem in fns if elem.endswith('.npz')]
    rows, cols = [], []
    for i in rc:
        j = i.split(os.sep)[-1].split('.npz')[0]
        x, y = j.split('_')
        rows.append(x)
        cols.append(y)
    for i in rows:
        for j in cols:
            z = sparse.csr_matrix(shape, dtype='float32')
            for k in rows:
                xn = tmp_path + i + '_' + k + '.npz'
                yn = tmp_path + k + '_' + j + '.npz'
                try:
                    x = load_matrix(xn, load=False)
                    y = load_matrix(yn, load=False)
                except:
                    continue
                z += x * y

            zn = tmp_path + i + '_' + j + '_new'
            sparse.save_npz(zn, z)

    # rename
    for i in xy:
        os.system('mv %s_new %s' % (i, i))


def mul1(qry, shape=(10**8, 10**8), tmp_path=None, xy=[], load=False):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    fns = os.listdir(tmp_path)
    if not xy:
        xy = [elem.split('.npz')[0].split('_')
                         for elem in fns if elem.endswith('.npz')]
        # xy = list(set(map(int, xy)))
        # print xy
        xy = sum(xy, [])
        xy = list(set(xy))
        xy.sort(key=lambda x: int(x))
    else:
        xy = map(str, xy)
    for i in xy:
        for j in xy:
            z = sparse.csr_matrix(shape, dtype='float32')
            for k in xy:
                xn = tmp_path + '/' + i + '_' + k + '.npz'
                x = load_matrix(xn, load=load)
                if i != j:
                    yn = tmp_path + '/' + k + '_' + j + '.npz'
                    y = load_matrix(yn, load=load)
                else:
                    y = x
                print 'current', (i, k), (k, j), x.shape, y.shape, z.shape
                z += x * y

            zn = tmp_path + '/' + i + '_' + j + '_new'
            sparse.save_npz(zn, z)

    # rename
    for i in xy:
        for j in xy:
            k = i + '_' + j
            os.system('mv %s_new.npz %s.npz' % (k, k))


def mul2(qry, shape=(10**8, 10**8), tmp_path=None, xy=[], load=False):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    fns = os.listdir(tmp_path)
    if not xy:
        xy = [elem.split('.npz')[0].split('_')
                         for elem in fns if elem.endswith('.npz')]
        # xy = list(set(map(int, xy)))
        # print xy
        xy = sum(xy, [])
        xy = list(set(xy))
        xy.sort(key=lambda x: int(x))
    else:
        xy = map(str, xy)
    for i in xy:
        for k in xy:
            xn = tmp_path + '/' + i + '_' + k + '.npz'
            x = load_matrix(xn, load=load)
            for j in xy:
                if i != j:
                    yn = tmp_path + '/' + k + '_' + j + '.npz'
                    y = load_matrix(yn, load=load)
                else:
                    y = x

                zn = tmp_path + '/' + i + '_' + j + '_new'
                try:
                    z = load_matrix(zn + '.npz', load=True)
                except:
                    z = sparse.csr_matrix(shape, dtype='float32')

                print 'current', (i, k), (k, j), x.shape, y.shape, z.shape
                z += x * y
                sparse.save_npz(zn, z)

    # rename
    for i in xy:
        for j in xy:
            k = i + '_' + j
            os.system('mv %s_new.npz %s.npz' % (k, k))


def mul3(qry, shape=(10**8, 10**8), tmp_path=None, xy=[], load=False):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    fns = os.listdir(tmp_path)
    if not xy:
        xy = [elem.split('.npz')[0].split('_')
                         for elem in fns if elem.endswith('.npz')]
        # xy = list(set(map(int, xy)))
        # print xy
        xy = sum(xy, [])
        xy = list(set(xy))
        xy.sort(key=lambda x: int(x))
    else:
        xy = map(str, xy)

    for i in xy:
        # get row
        xs = []
        for idx in xy:
            xn = tmp_path + '/' + i + '_' + idx + '.npz'
            try:
                x = load_matrix(xn, load=load)
            except:
                x = None

            print 'loading x', x.shape
            # raise SystemExit()
            xs.append(x)

        for j in xy:
            # get col
            ys = []
            for idx in xy:
                if idx == i:
                    y = xs[int(j)]
                else:
                    yn = tmp_path + '/' + idx + '_' + j + '.npz'
                    try:
                        y = load_matrix(yn, load=load)
                    except:
                        y = None

                print 'loading y', y.shape
                ys.append(y)

            Z = sparse.csr_matrix(shape, dtype='float32')

            for X, Y in zip(xs, ys):
                try:
                    # Z += X * Y

                    start = time()
                    tmp = X * Y
                    print 'time usage', i, j, time() - start

                    Z += tmp
                    del tmp
                    gc.collect()

                except:
                    continue

            zn = tmp_path + '/' + i + '_' + j + '_new'
            sparse.save_npz(zn, Z)

    # rename
    for i in xy:
        for j in xy:
            k = tmp_path + '/' + i + '_' + j
            os.system('mv %s_new.npz %s.npz' % (k, k))


def mul4(qry, shape=(10**8, 10**8), tmp_path=None, xy=[], load=False):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    fns = os.listdir(tmp_path)
    if not xy:
        xy = [elem.split('.npz')[0].split('_')
                         for elem in fns if elem.endswith('.npz')]
        # xy = list(set(map(int, xy)))
        # print xy
        xy = sum(xy, [])
        xy = list(set(xy))
        xy.sort(key=lambda x: int(x))
    else:
        xy = map(str, xy)

    row_sum = np.zeros(shape[0], dtype='float32')
    for i in xy:
        # get row
        xs = []
        for idx in xy:
            xn = tmp_path + '/' + i + '_' + idx + '.npz'
            try:
                x = load_matrix(xn, load=load)
            except:
                x = None

            print 'loading x', x.shape
            # raise SystemExit()
            xs.append(x)

        for j in xy:
            # get col
            ys = []
            for idx in xy:
                if idx == i:
                    y = xs[int(j)]
                else:
                    yn = tmp_path + '/' + idx + '_' + j + '.npz'
                    try:
                        y = load_matrix(yn, load=load)
                    except:
                        y = None

                print 'loading y', y.shape
                ys.append(y)

            Z = sparse.csr_matrix(shape, dtype='float32')

            for X, Y in zip(xs, ys):
                try:
                    # Z += X * Y

                    start = time()
                    tmp = X * Y
                    print 'time usage', i, j, time() - start

                    Z += tmp
                    del tmp
                    gc.collect()

                except:
                    continue

            zn = tmp_path + '/' + i + '_' + j + '_new'
            sparse.save_npz(zn, Z)
            row_sum += Z.sum(0)

    # rename
    for i in xy:
        for j in xy:
            k = tmp_path + '/' + i + '_' + j
            os.system('mv %s_new.npz %s.npz' % (k, k))

    return row_sum


def mul(qry, shape=(10**8, 10**8), tmp_path=None, csr=False):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    fns = [tmp_path + '/' +
        elem for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    row_sum = np.zeros(shape[0], dtype='float32')
    for i in fns:
        # get row
        Z = sparse.csr_matrix(shape, dtype='float32')
        x = load_matrix(i, shape=shape, csr=csr)
        for j in fns:
            # get col
            start = time()
            if i != j:
                y = load_matrix(j, shape=shape, csr=csr)
            else:
                y = x

            print 'loading time', i.split('/')[-1], j.split('/')[-1], time() - start

            start = time()
            tmp = x * y
            Z += tmp
            print 'multiple time', time() - start

            del tmp
            gc.collect()

        sparse.save_npz(i + '_new', Z)
        print 'saved', i
        row_sum += np.asarray(Z.sum(0))[0]

    # rename
    for i in fns:
        os.system('mv %s_new.npz %s' % (i, i))

    return row_sum


def expend0(qry, shape=(10**8, 10**8), tmp_path=None, csr=False, I=1.5, prune=1e-5, rtol=1e-5, atol=1e-8, check=False):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    err = float('+inf')
    fns = [tmp_path + '/' +
        elem for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    row_sum = np.zeros(shape[0], dtype='float32')
    for i in fns:
        # get row
        Z = sparse.csr_matrix(shape, dtype='float32')
        x = load_matrix(i, shape=shape, csr=csr)
        for j in fns:
            # get col
            start = time()
            if i != j:
                y = load_matrix(j, shape=shape, csr=csr)
            else:
                y = x

            print 'loading time', i.split('/')[-1], j.split('/')[-1], time() - start
            start = time()
            # tmp = x * y
            # tmp = np.dot(x, y)
            tmp = Pmul(x, y)
            Z += tmp
            print 'multiple time', time() - start

            del tmp
            gc.collect()

        Z.data **= I
        Z.data[Z.data < prune] = 0
        Z.eliminate_zeros()

        if check:
            err = min((abs(Z - x) - rtol * abs(x)).max(), err)

        sparse.save_npz(i + '_new', Z)
        print 'saved', i
        row_sum += np.asarray(Z.sum(0))[0]

    # rename
    for i in fns:
        os.system('mv %s_new.npz %s' % (i, i))

    if check:
        print 'current error', err
    cvg = err < atol
    return row_sum, fns, cvg


def expend2(qry, shape=(10**8, 10**8), tmp_path=None, csr=False, I=1.5, prune=1e-5, rtol=1e-5, atol=1e-8, check=False):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    err = None
    fns = [tmp_path + '/' +
        elem for elem in os.listdir(tmp_path) if elem.endswith('.npz')]

    num_set = [elem.split('.')[0].split('_')
                          for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    num_set = list(set(sum(num_set, [])))
    num_set.sort(key=lambda x: int(x))
    print 'num set is', num_set

    row_sum = np.zeros(shape[0], dtype='float32')
    for i in fns:
        # get row
        # Z = sparse.csr_matrix(shape, dtype='float32')
        Z_old = Z = None
        a, b = i.split(os.sep)[-1].split('.')[0].split('_')[:2]
        print 'current cell', a, b, num_set
        for j in num_set:

            start = time()
            xn = tmp_path + '/' + a + '_' + j + '.npz'
            yn = tmp_path + '/' + j + '_' + b + '.npz'
            try:
                x = load_matrix(xn, shape=shape, csr=csr)
            except:
                print 'can\'t load x', xn
                continue
            if xn != yn:
                try:
                    y = load_matrix(yn, shape=shape, csr=csr)
                except:
                    print 'can\'t load y', yn
                    continue
            else:
                y = x

            # print 'loading time', xn.split('/')[-1], yn.split('/')[-1],
            # time() - start
            start = time()
            # tmp = x * y
            # tmp = np.dot(x, y)
            # get old z
            if xn == i:
                Z_old = x
            elif yn == i:
                Z_old = y
            else:
                pass

            tmp = Pmul(x, y)
            try:
                Z += tmp
            except:
                Z = tmp
            print 'multiple time', time() - start, a, j, j, b

            del tmp
            gc.collect()

        Z.data **= I
        Z.data[Z.data < prune] = 0
        Z.eliminate_zeros()

        # if check:
        #    err = max((abs(Z-Z_old)-rtol * abs(Z_old)).max(), err)

        sparse.save_npz(i + '_new', Z)
        print 'saved', i
        row_sum += np.asarray(Z.sum(0))[0]

    # rename
    for i in fns:
        os.system('mv %s %s_old.npz' % (i, i))
        os.system('mv %s_new.npz %s' % (i, i))

    if check:
        print 'current error', err

    if err != None:
        cvg = err < atol
    else:
        cvg = False

    return row_sum, fns, cvg


def expend3(qry, shape=(10**8, 10**8), tmp_path=None, csr=False, I=1.5, prune=1e-5, cpu=1):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    err = None
    fns = [tmp_path + '/' +
        elem for elem in os.listdir(tmp_path) if elem.endswith('.npz')]

    num_set = [elem.split('.')[0].split('_')
                          for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    num_set = list(set(sum(num_set, [])))
    num_set.sort(key=lambda x: int(x))
    # print 'num set is', num_set

    row_sum = np.zeros(shape[0], dtype='float32')
    for i in fns:
        # get row
        # Z = sparse.csr_matrix(shape, dtype='float32')
        Z = None
        a, b = i.split(os.sep)[-1].split('.')[0].split('_')[:2]
        # print 'current cell', a, b, num_set
        for j in num_set:

            start = time()
            xn = tmp_path + '/' + a + '_' + j + '.npz'
            yn = tmp_path + '/' + j + '_' + b + '.npz'
            try:
                x = load_matrix(xn, shape=shape, csr=csr)
            except:
                # print 'can\'t load x', xn
                continue
            if xn != yn:
                try:
                    y = load_matrix(yn, shape=shape, csr=csr)
                except:
                    # print 'can\'t load y', yn
                    continue
            else:
                y = x

            # print 'loading time', xn.split('/')[-1], yn.split('/')[-1],
            # time() - start
            start = time()
            # tmp = x * y
            # tmp = np.dot(x, y)
            # get old z

            tmp = Pmul(x, y, cpu=cpu)
            try:
                Z += tmp
            except:
                Z = tmp
            # print 'multiple time', time() - start, a, j, j, b
            del tmp
            gc.collect()

        Z.data **= I
        Z.data[Z.data < prune] = 0
        Z.eliminate_zeros()

        sparse.save_npz(i + '_new', Z)
        # print 'saved', i
        row_sum += np.asarray(Z.sum(0))[0]

    # rename
    for i in fns:
        os.system('mv %s %s_old' % (i, i))
        os.system('mv %s_new.npz %s' % (i, i))

    return row_sum, fns

def sdot(x):
    xn, yn, shape, csr = x
    try:
        x = load_matrix(xn, shape=shape, csr=csr)
    except:
        return None
    if xn != yn:
        try:
            y = load_matrix(yn, shape=shape, csr=csr)
        except:
            return None 
    else:
        y = x

    z = x * y
    name = xn + '_tmp.npz'
    sparse.save_npz(name, z)
    return name


def expend(qry, shape=(10**8, 10**8), tmp_path=None, csr=True, I=1.5, prune=1e-5, cpu=1):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    err = None
    fns = [tmp_path + '/' +
        elem for elem in os.listdir(tmp_path) if elem.endswith('.npz')]

    num_set = [elem.split('.')[0].split('_')
                          for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    num_set = list(set(sum(num_set, [])))
    num_set.sort(key=lambda x: int(x))
    # print 'num set is', num_set

    row_sum = np.zeros(shape[0], dtype='float32')
    for i in fns:
        # get row
        a, b = i.split(os.sep)[-1].split('.')[0].split('_')[:2]
        xys = []
        for j in num_set:

            start = time()
            xn = tmp_path + '/' + a + '_' + j + '.npz'
            yn = tmp_path + '/' + j + '_' + b + '.npz'
            xys.append([xn, yn, shape, csr])

        if len(xys) > 1:
            zns = Parallel(n_jobs=cpu)(delayed(sdot)(elem) for elem in xys)
        elif len(xys) == 1:
            zns = [sdot(xys[0])]
        else:
            continue

        Z = None
        for zn in zns:
            if zn:
                tmp = load_matrix(zn, shape, csr)
                os.system('rm %s'%zn)
                try:
                    Z += tmp
                except:
                    Z = tmp

        Z.data **= I
        Z.data[Z.data < prune] = 0
        Z.eliminate_zeros()

        sparse.save_npz(i + '_new', Z)
        row_sum += np.asarray(Z.sum(0))[0]

    # rename
    for i in fns:
        os.system('mv %s %s_old' % (i, i))
        os.system('mv %s_new.npz %s' % (i, i))

    return row_sum, fns


# normalizatin
def norm0(qry, shape=(10**8, 10**8), tmp_path=None, row_sum=None):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    fns = os.listdir(tmp_path)
    if not xy:
        xy = [elem.split('.npz')[0].split('_')
                         for elem in fns if elem.endswith('.npz')]
        # xy = list(set(map(int, xy)))
        # print xy
        xy = sum(xy, [])
        xy = list(set(xy))
        xy.sort(key=lambda x: int(x))
    else:
        xy = map(str, xy)

    if row_sum == None:
        row_sum = np.zeros(shape[0], dtype='float32')

        for i in xy:
            for j in xy:
                xn = tmp_path + '/' + i + '_' + j + '.npz'
                try:
                    x = load_matrix(qry, load=True)
                    row_sum += x.sum(0)
                except:
                    continue

    for i in xy:
        for j in xy:
            xn = tmp_path + '/' + i + '_' + j + '.npz'
            try:
                x = load_matrix(qry, load=True)
                x.eliminate_zeros()
                x.data /= row_sum.take(x.indices, mode='clip')
                sparse.save_npz(xn, x)

            except:
                continue


def norm1(qry, shape=(10**8, 10**8), tmp_path=None, row_sum=None, csr=False):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    fns = [tmp_path + '/' +
        elem for elem in os.listdir(tmp_path) if elem.endswith('.npz')]

    if isinstance(row_sum, type(None)):
        row_sum = np.zeros(shape[0], dtype='float32')
        for i in fns:
            try:
                x = load_matrix(i, shape=shape, csr=csr)
                x = np.asarray(x.sum(0))[0]
                row_sum += x
            except:
                continue

    for i in fns:
        try:
            x = load_matrix(i, shape=shape, csr=csr)
            x.data /= row_sum.take(x.indices, mode='clip')
            sparse.save_npz(i + '_new', x)
        except:
            continue

    for i in fns:
        os.system('mv %s_new.npz %s' % (i, i))

    return fns


def norm2(qry, shape=(10**8, 10**8), tmp_path=None, row_sum=None, csr=False):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    fns = [tmp_path + '/' +
        elem for elem in os.listdir(tmp_path) if elem.endswith('.npz')]

    if isinstance(row_sum, type(None)):
        row_sum = np.zeros(shape[0], dtype='float32')
        for i in fns:
            try:
                x = load_matrix(i, shape=shape, csr=csr)
                x = np.asarray(x.sum(0))[0]
                row_sum += x
            except:
                continue

    for i in fns:
        try:
            x = load_matrix(i, shape=shape, csr=csr)
            x.data /= row_sum.take(x.indices, mode='clip')
            sparse.save_npz(i + '_new', x)
        except:
            continue

    for i in fns:
        os.system('mv %s_new.npz %s' % (i, i))

    return fns


def norm(qry, shape=(10**8, 10**8), tmp_path=None, row_sum=None, csr=False, rtol=1e-5, atol=1e-8, check=False):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    fns = [tmp_path + '/' +
        elem for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    if isinstance(row_sum, type(None)):
        row_sum = np.zeros(shape[0], dtype='float32')
        for i in fns:
            try:
                x = load_matrix(i, shape=shape, csr=csr)
                x = np.asarray(x.sum(0))[0]
                row_sum += x
            except:
                continue

    err = None
    for i in fns:
        try:
            x = load_matrix(i, shape=shape, csr=csr)
        except:
            continue
        try:
            x.data /= row_sum.take(x.indices, mode='clip')
        except:
            # print 'start norm3', check, x.data, row_sum.max()
            break

        if check:
            # print 'start norm4'
            x_old = load_matrix(i + '_old', shape=shape, csr=csr)
            # print 'start norm4 x x_old', abs(x - x_old).shape

            gap = abs(x - x_old) - abs(rtol * x_old)
            err = max(err, gap.max())
            # print 'check err is', err, i, i+'_old'

        sparse.save_npz(i + '_new', x)

    for i in fns:
        os.system('mv %s_new.npz %s' % (i, i))

    if err != None and err <= atol:
        cvg = True
    else:
        cvg = False

    return fns, cvg


# mcl algorithm
def mcl0(qry, tmp_path=None, xy=[], I=1.5, prune=1e-5, itr=100, rtol=1e-5, atol=1e-8, check=5):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    q2n = mat_split(qry)
    N = len(q2n)
    shape = (N, N)
    # norm
    print 'finish norm'
    # expension
    for i in xrange(itr):
        print 'iteration', i

        if i <= 0:
            print '1st row sum'
            fns = norm(qry, shape, tmp_path, csr=False)
        else:
            fns = norm(qry, shape, tmp_path, row_sum=row_sum, csr=True)

        if i > 0 and i % check == 0:
            row_sum, fns, cvg = expend(qry, shape, tmp_path, True, check=True)
        else:
            row_sum, fns, cvg = expend(qry, shape, tmp_path, True)

        if cvg:
            print 'yes, convergency'
            break


# merge two connected components array
@jit
def merge_connected(c0, c1):
    n0, a0 = c0
    n1, a1 = c1
    l0 = len(a0)
    l1 = len(a1)
    assert l0 == l1

    # sort by a1
    ht = np.zeros(n1, dtype='int')
    for i in a1:
        ht[i] += 1

    for i in xrange(1, n1):
        ht[i] += ht[i - 1]

    htc = ht.copy()
    s1 = np.empty(l1, dtype='int')
    for i in xrange(l1):
        x = a1[i]
        y = ht[x] - 1
        s1[y] = i
        ht[x] = y

    ht = htc
    # relabel a0
    visit = -np.ones(n0, dtype='int')
    a0_n = np.empty(l0, dtype='int')
    flag = 0
    total = 0
    for i in xrange(n1):
        if i <= 0:
            st, ed = 0, ht[i]
        else:
            st, ed = ht[i - 1], ht[i]

        total += ed - st
        # check current components has been visited
        c = -1
        for j in xrange(st, ed):
            cj = a0[s1[j]]

            if visit[cj] > -1:
                c = visit[cj]
                idx = s1[st:ed]
                a0_n[idx] = c
                visit[a0[idx]] = c
                break

        if c == -1:
            idx = s1[st:ed]
            a0_n[idx] = flag
            visit[a0[idx]] = flag
            flag += 1

        else:
            continue

    return flag, a0_n


def mcl(qry, tmp_path=None, xy=[], I=1.5, prune=1e-4, itr=100, rtol=1e-5, atol=1e-8, check=5, cpu=1, chunk=5*10**7):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    os.system('rm -rf %s' % tmp_path)

    q2n = mat_split(qry, chunk=chunk, cpu=cpu)
    N = len(q2n)
    prune = min(prune, 1e2 / N)
    shape = (N, N)
    # norm
    fns, cvg = norm(qry, shape, tmp_path, csr=False)
    # print 'finish norm', cvg
    # expension
    for i in xrange(itr):
        print '#iteration', i
        # row_sum, fns = expend(qry, shape, tmp_path, True, prune=prune,
        # cpu=cpu)
        row_sum, fns = expend(qry, shape, tmp_path, True, I, prune, cpu)

        if i > 0 and i % check == 0:
            fns, cvg = norm(qry, shape, tmp_path,
                            row_sum=row_sum, csr=True, check=True)
        else:
            fns, cvg = norm(qry, shape, tmp_path, row_sum=row_sum, csr=True)

        if cvg:
            # print 'yes, convergency'
            break

    # get connect components
    g = load_matrix(fns[0], shape, True)
    cs = csgraph.connected_components(g)
    for fn in fns[1:]:
        g = load_matrix(fn, shape, True)
        ci = csgraph.connected_components(g)
        cs = merge_connected(cs, ci)

    del g
    gc.collect()

    # print 'find components', cs
    groups = {}
    for k, v in q2n.iteritems():
        c = cs[1][v]
        try:
            groups[c].append(k)
        except:
            groups[c] = [k]

    del c
    gc.collect()
    for v in groups.itervalues():
        print '\t'.join(v)


# print the manual
def manual_print():
    print 'Usage:'
    print '    python %s -i foo.xyz -I 0.5 -a 8' % sys.argv[0]
    print 'Parameters:'
    print '  -i: tab-delimited file which contain 3 columns'
    print '  -I: inflation parameter for mcl'
    print '  -a: cpu number'
    print '  -b: chunk size. default value is 50000000'

if __name__ == '__main__':

    argv = sys.argv
    # recommand parameter:
    args = {'-i': '', '-I': '1.5', '-a': '2', '-b': '50000000'}

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
        qry, ifl, cpu, bch = args['-i'], float(eval(args['-I'])), int(eval(args['-a'])), int(eval(args['-b']))

    except:
        manual_print()
        raise SystemExit()

    # convert the relationship into numeric and split into small matrix
    # q2n, xy = mat_split(qry)
    # mul(qry, xy=xy, load=False)
    # mul(qry, load=False)
    # mul(qry, load=True)
    # q2n = mat_split(qry)
    # mul(qry, csr=False)
    mcl(qry, I=ifl, cpu=cpu, chunk=bch)

    # preprocess(qry)
    raise SystemExit()

    # qry = sys.argv[1]
    # refs = sys.argv[2:]
    mats = [elem for elem in os.listdir('./') if elem.endswith('.bin')]
    for i in mats:
        x = load_matrix(i)
        sparse.save_npz(i, x)
        os.system('rm %s'%i)

    mats = [elem for elem in os.listdir('./') if elem.endswith('.bin.npz')]
    qrys = [elem for elem in mats if 'row_' in elem]
    refs = [elem for elem in mats if 'col_' in elem]


    qrys.sort()
    refs.sort()

    for qry in qrys:
        gc.collect()

        start = time()
        x, y = map(sparse.load_npz, [qry, refs[0]])
        print 'loading', time() - start

        start = time()
        z = x * y
        print 'mul by', refs[0], time() - start

        for ref in refs[1:]:
            start = time()
            # y = load_matrix(ref)
            y = sparse.load_npz(ref)
            print 'loading', time() - start

            start = time()
            z += x * y
            print 'mul by', ref, time() - start

