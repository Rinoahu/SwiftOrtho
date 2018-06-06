#!usr/bin/env python
from numba import jit
import numpy as np
from scipy import sparse as sps
try:
    from numba import jit
except:
    jit = lambda x: x


#jit = lambda x: x


@jit
def resize(a, new_size):
    new = np.empty(new_size, a.dtype)
    new[:a.size] = a
    return new


@jit
def resize0(a, new_size):
    new = np.asarray(np.memmap('tmp.npy', mode='w+', shape=new_size, dtype=a.dtype), dtype=a.dtype)
    #new = np.empty(new_size, a.dtype)
    new[:a.size] = a
    return new



# csr matrix by matrix
@jit
def csrmm(xr, xc, x, yr, yc, y):

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



#csrmm_jit = jit(csrmm)

def csrmm_ez(a, b, use_jit=True):
    xr, xc, x = a.indptr, a.indices, a.data
    yr, yc, y = b.indptr, b.indices, b.data

    print 'a shape', a.shape, 'b shape', b.shape, 'yc size', yc[:10], yc.size, yc.max(), yc[-1], 'yr', yr.size, yr[:10]


    st = time()
    #if use_jit:
    #    zr, zc, z, flag = csrmm_jit(xr, xc, x, yr, yc, y)
    #else:
    #    zr, zc, z, flag = csrmm(xr, xc, x, yr, yc, y)

    zr, zc, z, flag = csrmm(xr, xc, x, yr, yc, y)

    print 'total operation', flag
    print 'csrmm cpu', time() - st
    print 'zr min', zr.min(), 'zc max', zr.max(), 'zr size', zr.size 
    print 'zc min', zc.min(), 'zc max', zc.max(), 'zc size', zc.size
    zmtx = sps.csr_matrix((z, zc, zr), shape=(a.shape[0], b.shape[1]))
    return zmtx


if __name__ == '__main__':
    from time import time
    import sys

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
    y0 = csrmm_ez(x, y)
    print 'my sparse', time() - st, y0.shape

    st = time()
    y1 = x * y
    print 'scipy csr', time() - st, y1.shape

    dif = y0 - y1
    print dif.max(), dif.min(), 'my nnz', y0.nnz, 'scipy nnz', y1.nnz
