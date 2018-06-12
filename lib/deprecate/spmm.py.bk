#!usr/bin/env python
from numba import jit
import numpy as np
from scipy import sparse as sps
try:
    from numba import jit, njit, prange
except:
    njit = jit = lambda x: x
    prange = xrange

from multiprocessing.pool import ThreadPool as Pool
from threading import Thread
from Queue import Queue

from time import time
import sys


#jit = lambda x: x


class worker(Thread):

    def __init__(self,func,args=()):
        super(worker, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None




class worker0(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs, Verbose)
        self._return = None
    def run(self):
        if self._Thread__target is not None:
            self._return = self._Thread__target(*self._Thread__args,
                                                **self._Thread__kwargs)
    def join(self):
        Thread.join(self)
        return self._return



@njit
def resize(a, new_size):
    new = np.empty(new_size, a.dtype)
    new[:a.size] = a
    return new


@njit
def resize_mmp(a, new_size):
    new = np.asarray(np.memmap('tmp.npy', mode='w+', shape=new_size, dtype=a.dtype), dtype=a.dtype)
    new[:a.size] = a
    return new



# csr matrix by matrix
# original version
@jit(fastmath=True, nogil=True)
def csrmm_ori(xr, xc, x, yr, yc, y):

    R = xr.shape[0]
    D = yr.shape[0]
    nnz = int(1. * x.size * y.size / (D-1))
    print 'nnz size', nnz
    # zr, zc, z = np.zeros(R, 'int32'), np.empty(nnz*5, 'int32'), np.empty(nnz*5, dtype=x.dtype)
    n_size = nnz
    zr, zc, z = np.zeros(R, xr.dtype), np.empty(n_size, xc.dtype), np.empty(n_size, dtype=x.dtype)
    data = np.zeros(D-1, dtype=x.dtype)
    #print 'zr init', zr[:5]

    # hash table
    visit = np.zeros(yr.size, 'int8')
    index = np.zeros(yr.size, yr.dtype)
    flag = 0
    zptr = 0
    for i in xrange(R-1):

        # get ith row of a
        kst, ked = xr[i], xr[i + 1]
        if kst == ked:
            zr[i+1] = zr[i]
            continue

        ks = 0
        nz = 0
        for k in xrange(kst, ked):
            x_col, x_val = xc[k], x[k]

            # get row of b
            jst, jed = yr[x_col], yr[x_col + 1]
            if jst == jed:
                continue

            nz += jed - jst
            for j in xrange(jst, jed):
                y_col, y_val = yc[j], y[j]
                data[y_col] += x_val * y_val
                            
                if visit[y_col] == 0:
                    visit[y_col] = 1
                    index[ks] = y_col
                    ks += 1
                    flag += 3
                #nz += 1
                flag += 3
            flag += 2


        zend = zr[i] + nz
        if zend > n_size:
            n_size += nnz
            #print('resize sparse matrix', n_size)
            zc = resize(zc, n_size)
            z = resize(z, n_size)
            flag += 2

        for pt in xrange(ks):
            idx = index[pt]
            #mx_col = max(mx_col, idx)
            val = data[idx]
            visit[idx] = 0
            if val > 0:
                zc[zptr], z[zptr] = idx, val
                zptr += 1
                data[idx] = 0
                flag += 5

            flag += 1

        zr[i+1] = zptr

    return zr, zc[:zptr], z[:zptr], flag
    #zmtx = sps.csr_matrix((z[:zptr], zc[:zptr], zr), shape=(a.shape[0], b.shape[1]))
    #return zmtx


# memory save version
#@njit(parallel=True, fastmath=True)
@njit(fastmath=True, nogil=True)
def csrmm_msav(xr, xc, x, yr, yc, y):

    R = xr.shape[0]
    D = yr.shape[0]
    chk = x.size + y.size
    nnz = chk
    print 'nnz size', nnz
    # zr, zc, z = np.zeros(R, 'int32'), np.empty(nnz*5, 'int32'), np.empty(nnz*5, dtype=x.dtype)
    zr, zc, z = np.zeros(R, xr.dtype), np.empty(nnz, xc.dtype), np.empty(nnz, dtype=x.dtype)
    data = np.zeros(D-1, dtype=x.dtype)
    #print 'zr init', zr[:5]

    # hash table
    #visit = np.zeros(yr.size, 'int8')
    index = np.zeros(yr.size, yr.dtype)
    flag = 0
    zptr = 0
    for i in xrange(R-1):

        # get ith row of a
        kst, ked = xr[i], xr[i + 1]
        if kst == ked:
            zr[i+1] = zr[i]
            continue

        ks = 0
        nz = 0
        for k in xrange(kst, ked):
            x_col, x_val = xc[k], x[k]

            # get row of b
            jst, jed = yr[x_col], yr[x_col + 1]
            if jst == jed:
                continue

            nz += jed - jst
            flag += 2

            for j in xrange(jst, jed):
            #for j in prange(jst, jed):
                y_col, y_val = yc[j], y[j]
                y_col_val = data[y_col] + x_val * y_val
                if y_col_val != 0:
                    index[ks] = y_col
                    ks += 1
                    flag += 2

                data[y_col] = y_col_val
                flag += 3


        zend = zr[i] + nz
        if zend > nnz:
            nnz += chk
            #print('resize sparse matrix', n_size)
            zc = resize(zc, nnz)
            z = resize(z, nnz)
            flag += 2

        for pt in xrange(ks):
        #for pt in prange(ks):
            y_col = index[pt]
            #mx_col = max(mx_col, idx)
            y_col_val = data[y_col]
            if y_col_val != 0:
                zc[zptr], z[zptr] = y_col, y_col_val
                zptr += 1
                data[y_col] = 0
                flag += 3

            flag += 1


        zr[i+1] = zptr

    return zr, zc[:zptr], z[:zptr], flag
    #zmtx = sps.csr_matrix((z[:zptr], zc[:zptr], zr), shape=(a.shape[0], b.shape[1]))
    #return zmtx

# parallel version of csrmm
@njit(fastmath=True, nogil=True)
def csrmm_sp(Xr, Xc, X, yr, yc, y, xrst, xred, cpu=1):

    xr = np.empty(xred+1-xrst, Xr.dtype)
    xr[:] = Xr[xrst:xred+1]
    xr -= xr[0]
    xcst, xced = Xr[xrst], Xr[xred]
    xc = Xc[xcst: xced]
    x = X[xcst: xced]
    #print 'xrst %d xred %d xcst %d xced %d'%(xrst, xred, xcst, xced)

    R = xr.shape[0]
    D = yr.shape[0]
    chk = x.size + y.size
    nnz = chk
    #print 'nnz size %d %d %d %d'%(nnz, x.size, y.size, X.size)
    # zr, zc, z = np.zeros(R, 'int32'), np.empty(nnz*5, 'int32'), np.empty(nnz*5, dtype=x.dtype)
    zr, zc, z = np.zeros(R, xr.dtype), np.empty(nnz, xc.dtype), np.empty(nnz, dtype=x.dtype)
    data = np.zeros(D-1, dtype=x.dtype)
    #print 'zr init', zr[:5]

    # hash table
    #visit = np.zeros(yr.size, 'int8')
    chk1 = yr.size // cpu
    nnz1 = chk1
    index = np.zeros(nnz1, yr.dtype)
    flag = 0
    zptr = 0
    for i in xrange(R-1):

        # get ith row of a
        kst, ked = xr[i], xr[i + 1]
        if kst == ked:
            zr[i+1] = zr[i]
            continue

        ks = 0
        nz = 0
        for k in xrange(kst, ked):
            x_col, x_val = xc[k], x[k]

            # get row of b
            jst, jed = yr[x_col], yr[x_col + 1]
            if jst == jed:
                continue

            nz += jed - jst
            flag += 2

            for j in xrange(jst, jed):
            #for j in prange(jst, jed):
                y_col, y_val = yc[j], y[j]
                y_col_val = data[y_col] + x_val * y_val
                if y_col_val != 0:
                    if ks >= nnz1:
                        nnz1 += chk1
                        index = resize(index, nnz1)

                    index[ks] = y_col
                    ks += 1
                    flag += 2

                data[y_col] = y_col_val
                flag += 3


        zend = zr[i] + nz
        if zend > nnz:
            nnz += chk
            #print('resize sparse matrix', n_size)
            zc = resize(zc, nnz)
            z = resize(z, nnz)
            flag += 2

        for pt in xrange(ks):
        #for pt in prange(ks):
            y_col = index[pt]
            #mx_col = max(mx_col, idx)
            y_col_val = data[y_col]
            if y_col_val != 0:
                zc[zptr], z[zptr] = y_col, y_col_val
                zptr += 1
                data[y_col] = 0
                flag += 3

            flag += 1


        zr[i+1] = zptr

    #zr += xrst
    #print 'retunr value', zr
    return zr, zc[:zptr], z[:zptr], flag

#@jit(nogil=True)
#def csrmm_sp_wrapper(elem):
#    Xr, Xc, X, yr, yc, y, xrst, xred = elem
#    zr, zc, z, flag = csrmm_sp(Xr, Xc, X, yr, yc, y, xrst, xred)
#    print 'get_value'
#    #return zr, zc, z, flag
#    return sps.csr_matrix((z, zc, zr), dtype=z.dtype)



#csrmm_jit = jit(csrmm)

def csrmm_ez(a, b, mm='msav', cpu=1):
    xr, xc, x = a.indptr, a.indices, a.data
    yr, yc, y = b.indptr, b.indices, b.data

    #print 'a shape', a.shape, 'b shape', b.shape, 'yc size', yc[:10], yc.size, yc.max(), yc[-1], 'yr', yr.size, yr[:10]
    print 'a nnz', a.nnz, 'b nnz', b.nnz

    st = time()
    #if use_jit:
    #    zr, zc, z, flag = csrmm_jit(xr, xc, x, yr, yc, y)
    #else:
    #    zr, zc, z, flag = csrmm(xr, xc, x, yr, yc, y)
    if cpu > 1:
        csrmm = csrmm_sp
    elif mm == 'msav':
        csrmm = csrmm_msav
    elif mm == 'ori':
        csrmm = csrmm_ori
    else:
        raise SystemExit()

    if cpu <= 1:
        zr, zc, z, flag = csrmm(xr, xc, x, yr, yc, y)
        zmtx = sps.csr_matrix((z, zc, zr), shape=(a.shape[0], b.shape[1]))
    else:
        print 'using threads'
        N, D = a.shape
        step = N // cpu + 1
        threads = []
        for i in xrange(0, N, step):
            start, end = i, min(i+step, N)
            t = worker(csrmm_sp, (xr, xc, x, yr, yc, y, start, end, cpu))
            t.start()
            threads.append(t)

        res = []
        #offset = 0
        #for t in threads:
        flag = 0
        for i in xrange(0, N, step):
            start, end = i, min(i+step, N)
            t = threads[i//step]
            t.join()
            zr, zc, z, flag0 = t.get_result()
            new_shape = (end-start, b.shape[1])
            #print 'new shape', new_shape, z
            res.append(sps.csr_matrix((z, zc, zr), shape=new_shape, dtype=z.dtype))
            #print 'res', res
            flag += flag0

        #print res
        zmtx = sps.vstack(res)
        #paras = []
        #for i in xrange(0, N, step):
        #    start, end = i, min(i+step, N)
        #    paras.append([xr, xc, x, yr, yc, y, start, end])

        
        #pool = Pool(cpu)
        #results = pool.map(csrmm_sp_wrapper, paras)
        #results = map(csrmm_sp_wrapper, paras)


       #print 'threads is', threads
        #flag = sum([elem[-1] for elem in threads])

    print 'total operation', flag
    print 'csrmm cpu', time() - st
    #print 'zr min', zr.min(), 'zc max', zr.max(), 'zr size', zr.size 
    #print 'zc min', zc.min(), 'zc max', zc.max(), 'zc size', zc.size
    #zmtx = sps.csr_matrix((z, zc, zr), shape=(a.shape[0], b.shape[1]))
    return zmtx


# csr by csc
@jit
def csr_by_csc(xr, xc, x, yc, yr, y):

    R = xr.size
    C = yc.size
    chk = x.size + y.size
    nnz = chk
    print 'nnz size', nnz
    # zr, zc, z = np.zeros(R, 'int32'), np.empty(nnz*5, 'int32'), np.empty(nnz*5, dtype=x.dtype)
    zr, zc, z = np.zeros(R, xr.dtype), np.empty(nnz, xc.dtype), np.empty(nnz, dtype=x.dtype)

    ptr = 0
    flag = 0
    for i in xrange(R-1):

        # get ith row of a
        rst, red = xr[i], xr[i + 1]
        #N = red - rst
        if rst == red:
            zr[i+1] = ptr
            #flag += 1
            continue
        #flag += 2
        for j in xrange(C-1):
            # get row of b
            cst, ced = yc[j], yc[j+1]
            #flag += 2
            if cst == ced:
                continue

            #M = ced - cst
            #D = min(N, M)
            #D = red-rst, ced-cst)
            #for k in xrange(D):

            p0, p1 = rst, cst
            val = 0
            #flag += 3
            while p0 < red and p1 < ced:
            #for d in xrange(D):
                if xc[p0] < yr[p1]:
                    p0 += 1
                elif xc[p0] > yr[p1]:
                    p1 += 1
                else:
                    val += x[p0] * y[p1]
                    p0 += 1
                    p1 += 1
                flag += 1
            if val > 0:
                if ptr < nnz:
                    z[ptr] = val
                    zc[ptr] = j
                    ptr += 1
                else:
                    nnz += chk
                    z = resize(z, nnz)
                    zc = resize(zc, nnz)
                    z[ptr] = val
                    zc[ptr] = j
                    ptr += 1

                flag += 1
        zr[i+1] = ptr

    print 'final_ptr', ptr
    return zr, zc[:ptr], z[:ptr], flag
    #zmtx = sps.csr_matrix((z[:zptr], zc[:zptr], zr), shape=(a.shape[0], b.shape[1]))
    #return zmtx


# csr by csc version, slow
def cscmm_ez(a, c, use_jit=True):
    xr, xc, x = a.indptr, a.indices, a.data
    b = c.tocsc()
    yc, yr, y = b.indptr, b.indices, b.data

    print 'a shape', a.shape, 'b shape', b.shape, 'yc size', yc[:10], yc.size, yc.max(), yc[-1], 'yr', yr.size, yr[:10]


    st = time()
    #if use_jit:
    #    zr, zc, z, flag = csrmm_jit(xr, xc, x, yr, yc, y)
    #else:
    #    zr, zc, z, flag = csrmm(xr, xc, x, yr, yc, y)

    zr, zc, z, flag = csr_by_csc(xr, xc, x, yc, yr, y)

    print 'total operation', flag
    print 'csrmm cpu', time() - st
    print 'zr min', zr.min(), 'zc max', zr.max(), 'zr size', zr.size 
    print 'zc min', zc.min(), 'zc max', zc.max(), 'zc size', zc.size
    zmtx = sps.csr_matrix((z, zc, zr), shape=(a.shape[0], b.shape[1]))
    return zmtx






if __name__ == '__main__':
    st = time()
    try:
        a, b = sys.argv[1:3]
        x, y = map(sps.load_npz, [a, b])
    except:
        N = int(eval(sys.argv[1]))
        x = sps.random(N, N, 2./N, format='csr')
        row, col = x.nonzero()
        row += N//2
        col += N//2
        x = sps.csr_matrix((x.data,(row, col)), shape=(2*N, 2*N))
        y = x
    #sps.save_npz('tmp.npz', x)
    print 'random matrix', time() - st

    #small = sps.random(100, 100, .01, format='csr')
    #small_xy = csrmm_ez(small, small)

    st = time()
    y0 = csrmm_ez(x, y, 'msav')
    print 'my sparse msav', time() - st, y0.shape
    print ''

    st = time()
    y1 = csrmm_ez(x, y, 'ori')
    print 'my sparse ori', time() - st, y0.shape
    print ''

    st = time()
    y2 = csrmm_ez(x, y, 'msav', cpu=4)
    print 'my sparse parallel', time() - st, y0.shape
    print ''

    st = time()
    y3 = x * y
    print 'scipy csr', time() - st, y1.shape
    print ''

    dif = y2 - y3
    print dif.max(), dif.min(), 'my fast parallel nnz', y2.nnz, 'scipy nnz', y3.nnz
