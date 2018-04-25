#!usr/bin/env python
import numpy as np
from scipy import sparse
import sys
from time import time
import os
import gc
from struct import pack, unpack

# given a pairwise relationship, this function will convert the qid, sid into numbers
# and split these relationships into small file
def mat_split(qry, shape=10**7, step=2*10**5, tmp_path=None):
    #_os0 = [open('row_%d.bin'%elem, 'wb') for elem in xrange(6*10**6//step)]
    #_os1 = [open('col_%d.bin'%elem, 'wb') for elem in xrange(6*10**6//step)]
    # build the tmp dir
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    os.system('mkdir -p %s'%tmp_path)

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

        out = pack('fff', *[x,y,z])

        xi, yi = x // step, y // step
        try:
            _ox = _oxs[xi]
            _ox.write(out)

        except:
            _o = open(tmp_path + '/%d_row.bin'%xi, 'wb')
            n, m = len(_oxs), xi + 1
            if n < m:
                _oxs.extend([None]* (m - n))

            _oxs[xi] = _o
            _ox = _oxs[xi]
            #print 'xi is', _oxs[xi], _o
            _ox.write(out)

        try:
            _oy = _oys[yi]
            _oy.write(out)
        except:
            _o = open(tmp_path + '/%d_col.bin'%yi, 'wb')
            n, m = len(_oys), yi + 1
            if n < m:
                _oys.extend([None]* (m - n))

            _oys[yi] = _o
            _oy = _oys[yi]
            _oy.write(out)


        # sym
        x, y = y, x
        out = pack('fff', *[x,y,z])
        xi, yi = x // step, y // step

        try:
            _ox = _oxs[xi]
            _ox.write(out)

        except:
            _o = open(tmp_path + '/%d_row.bin'%xi, 'wb')
            n, m = len(_oxs), xi + 1
            if n < m:
                _oxs.extend([None]* (m - n))

            _oxs[xi] = _o
            _ox = _oxs[xi]
            #print 'xi is', _oxs[xi], _o
            _ox.write(out)

        try:
            _oy = _oys[yi]
            _oy.write(out)
        except:
            _o = open(tmp_path + '/%d_col.bin'%yi, 'wb')
            n, m = len(_oys), yi + 1
            if n < m:
                _oys.extend([None]* (m - n))

            _oys[yi] = _o
            _oy = _oys[yi]
            _oy.write(out)

        if eye[x] < z:
            eye[x] = z
        if eye[y] < z:
            eye[y] = z

    # set eye of matrix:
    for i in xrange(0, q2n, step):
        zs = q2n[i:i+step]
        xyzs = [[x,x,z] for x, z in zip(xrange(i, i+step), zs)]
        xyzs = sum(xyzs, [])
        out = pack('f'*len(xyzs), *xyzs)
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


# load sparse matrix from disk
def load_matrix(qry, shape=(10**9, 10**9), cvt=True):
    if cvt:
        n0 = np.memmap(qry, mode='r+', dtype='float32').shape[0] / 3
        dat0 =  np.memmap(qry, mode='r+', dtype='float32', shape=(n0, 3))
        x0, x1, x2 = dat0[:, 0], dat0[:, 1], dat0[:, 2]
        x = sparse.coo_matrix((x2, (x0, x1)), shape = shape, dtype='float32').tocsr()
    else:
        x = sparse.load_npz(qry)

    return x


# split row block and col block into row_col block 
def preprocess(qry, tmp_path=None):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'
    fns = os.listdir(tmp_path)
    rows = [elem for elem in fns if elem.endswith('_row.bin')]
    cols = [elem for elem in fns if elem.endswith('_col.bin')]

    for i in rows:
        xn = tmp_path+'/' + i
        x = load_matrix(xn)
        #x += x.transpose()
        for j in cols:
            yn = tmp_path+'/'+j
            y = load_matrix(yn)
            #y += y.transpose()
            z = x * y
            xi = i.split(os.sep)[-1].split('_row')[0]
            yj = j.split(os.sep)[-1].split('_col')[0]
            ij = xi +'_' + yj
            sparse.save_npz(tmp_path+'/'+ij, z)
            del y, z
            gc.collect()

        # remove x
        del x
        gc.collect()
        os.system('rm %s'%xn)

# matrix mul on small blocks
def mul(qry, shape=(10**7, 10**7), tmp_path=None):
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
                x = load_matrix(xn)
                yn = tmp_path + k + '_' + j + '.npz'
                y = load_matrix(yn)
                z += x * y

            zn = tmp_path + i + '_' + j + '.npz_new'
            sparse.save_npz(zn, z)

    # rename
    for i in rc:
        os.system('mv %s_new %s'%(i, i))


# normalizatin
def norm(qry, tmp_path=None):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'






if __name__ == '__main__':

    if len(sys.argv[1:]) < 1:
        print 'python this.py foo.abc|foo.abcd'

    qry = sys.argv[1]

    # convert the relationship into numeric and split into small matrix
    #q2n = mat_split(qry)

    preprocess(qry)
    raise SystemExit()

    #qry = sys.argv[1]
    #refs = sys.argv[2:]
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
            #y = load_matrix(ref)
            y = sparse.load_npz(ref)
            print 'loading', time() - start

            start = time()
            z += x * y
            print 'mul by', ref, time() - start

