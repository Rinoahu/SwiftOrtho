#!usr/bin/env python
# Clustering by fast search and find of density peak algorithm

import sys
from collections import Counter
from heapq import merge
from math import exp
import cPickle as pkl

try:
    qry = sys.argv[1]
except:
    print 'python this.py foo.xyz'
    raise SystemExit()

# the rho density
rho = {}

# find the dc
f = open(qry, 'r')
d = []
dtmp = []
N = 0
for i in f:
    x, y, z = i[:-1].split('\t')[:3]
    if x == y:
        continue

    z = float(z)
    dtmp.append(z)
    if len(dtmp) >= 10**8:
        d = list(merge(d, dtmp))[:10**8]
        dtmp = []
    rho[x] = [0, 0]
    rho[y] = [0, 0]
    N += 1

if len(dtmp) > 0:
    d = list(merge(d, dtmp))[:10**8]

f.close()

di = int(.2 * N)
dc = len(d) < di and d[di] or d[-1]


# compute the rho
f = open(qry, 'r')
for i in f:
    x, y, z = i[:-1].split('\t')[:3]
    if x == y:
        continue

    z = float(z)
    gauss = exp(-(z/dc)**2)
    rho[x] += gauss
    rho[y] += gauss


f.close()


# find the minimum distance of i to j whose rho is higher than i

delta = {}
f = open(qry, 'r')
for i in f:
    x, y, z = i[:-1].split('\t')[:3]
    if x == y:
        continue

    xr, yr = rho[x], rho[y]
    if xr < yr:
        if x not in delta or delta[x] > z:
            delta[x] = z
    else:
        if y not in delta or delta[y] > z:
            delta[y] = z

f.close()

