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


mean = lambda x: sum(x) * 1. / len(x)
def sd(x):
        n = len(x)
        mu = mean(x)
        var = float(reduce(lambda a, b: a + b, map(lambda a: (a - mu) ** 2., x))) / n
        std = var ** .5
        return std



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
        dtmp.sort()
        d = list(merge(d, dtmp))[:10**8]
        dtmp = []
    rho[x] = 0
    rho[y] = 0
    N += 1

if len(dtmp) > 0:
    dtmp.sort()
    d = list(merge(d, dtmp))[:10**8]

f.close()

di = int(.02 * N)
#print 'di', di, N
dc = len(d) < di and d[-di] or d[-1]


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

# find i's nearest point whose density is larger than i's
nn = {}

f = open(qry, 'r')
for i in f:
    x, y, z = i[:-1].split('\t')[:3]
    if x == y:
        continue

    z = float(z)
    xr, yr = rho[x], rho[y]

    if xr < yr:
        if x not in delta or delta[x] > z:
            delta[x] = z
            nn[x] = y
    if yr < xr:
        if y not in delta or delta[y] > z:
            delta[y] = z
            nn[y] = x

f.close()

print 'nn size', len(nn)
rnn = {}
while nn:
    k, v = nn.popitem()
    #if k == 'isolate_virus|650627109|650627007|' or v == 'isolate_virus|650627109|650627007|':
    #    print 'max_id', k, v
    try:
        rnn[v].append(k)
    except:
        rnn[v] = [k]

#print 'rnn', len(rnn) + sum(map(len, rnn.values()))

mx_delta = max(delta.itervalues())

mx_rho = -1
mx_id = None
for i in rho:
    r = rho[i]
    if mx_rho < r:
        mx_rho = r
        mx_id = i

delta[mx_id] = mx_delta
#nn[x] = x
#print 'x is', x, len(delta), len(nn)

#f = open(qry, 'r')
#for i in f:
#    x, y, z = i[:-1].split('\t')[:3]
#    if x == y:
#        continue
#
#    z = float(z)
#    xr, yr = rho[x], rho[y]
#    if xr == mx_rho or yr == mx_rho:
#        print 'max_rho is', x, y



# find clusters
rho_min = mean(rho.values())
deltamin = sd(delta.values())
print 'rho_min', rho_min, 'delta_min', deltamin
cl = {}
flag = 0
flag2 = 0
for i in delta:
    if rho[i] > rho_min and delta[i] > deltamin:
        cl[i] = flag
        flag += 1
        #continue

nn = {}
f = open(qry, 'r')
for i in f:
    x, y, z = i[:-1].split('\t')[:3]
    if x == y:
        continue

    z = float(z)
    xr, yr = rho[x], rho[y]

    if xr < yr and y in cl and x not in cl:
        try:
            if z < nn[x][1]:
                nn[x] = [cl[y], z]
        except:
            nn[x] = [cl[y], z]

    if yr < xr and x in cl and y not in cl:
        try:
            if z < nn[y][1]:
                nn[y] = [cl[x], z]
        except:
            nn[y] = [cl[x], z]


print 'total_nn_cl', len(cl), len(nn)




#for i in delta:
kys = cl.keys()
for i in kys:
    if 1:
        c = cl[i]
        # assign neigbor
        try:
            stack = rnn[i]
        except:
            continue

        visit = set()
        #stack = set([i])
        while stack:
            x = stack.pop()
            if x not in visit:
                visit.add(x)
                cl[x] = c
                nbs = rnn.get(x, [])
                stack.extend(rnn.get(x, []))
                #stack.extend([elem for elem in nbs if elem not in cl])
        #if i == mx_id:
        #    print 'length', len(visit)
        #visit.add(i)
        #print i, len(visit)
        #flag2 += len(visit)
        #flag += 1

print 'flag is', flag, len(cl)

#for i in rho:
#    if i not in cl:
#        cl[i] = flag
#        flag += 1
#print 'total flag', flag, flag2, len(cl), sum(map(len, rnn.values()))


# find halo point
bord_rho = {}
f = open(qry, 'r')
for i in f:
    x, y, z = i[:-1].split('\t')[:3]
    if x == y:
        continue

    z = float(z)
    xr, yr = rho[x], rho[y]

    #if x not in cl or y not in cl:
    #    continue
    cx, cy = cl.get(x, -1), cl.get(y, -1)

    if cx == -1 or cy == -1:
        continue

    if cx != cy and z <= dc:
        rho_avg = (xr+yr) / 2.
        if rho_avg > bord_rho.get(cx, 0):
            bord_rho[cx] = rho_avg
        if rho_avg > bord_rho.get(cy, 0):
            bord_rho[cy] = rho_avg

#halo = {}
#for i in rho:
#    if rho[i] < bord_rho.get(i, 0):
#        #halo[i] = -1
#        cl[i] = -1
#for i in cl:
for i in rho:
    c = cl.get(i, -1)
    if c == -1:
        continue
    rhoi, bdi = rho.get(i, 0), bord_rho.get(c, 0)
    print x, 'cluster', c, rhoi, bdi, rhoi < bdi and 'halo' or 'core', len(cl), len(rho), len(delta)
   

raise SystemExit()


# get cluster of point i
def assign_cluster(i, cl, nn):
    #if i in cl:
    #    return cl[i]
    #else:
    #    cl[i] = assign_cluster(nn[i], cl, nn)
    #    return cl[i]
    visit = set()
    c = -1
    stack = set([i])
    #print 'i is', i, nn.keys()[0], len(nn)
    while stack:
        x = stack.pop()
        if x in visit:
            continue
        #if x in cl:
        #    return x
        #else:
        #    #stack.append(x)
        #    stack.add(x)
        #    y = nn[x]
        #    #stack.append(y)
        #    stack.add(y)
        else:
            visit.add(x)
        if x in cl:
            c = cl[x]
            break
        else:
            #visit.add(x)
            try:
                y = nn[x]
                stack.add(y)
            except:
                continue

    if c > -1:
        print i, c, 'cl length', len(cl)
        for x in visit:
            cl[x] = c

# assign other point to cluster
for i in delta:
    break
    if i not in cl:
        assign_cluster(i, cl, nn)
    else:
        continue



# filter the noise





#print cl
print 'cl is', cl.values().count(-1), min(cl.values()), len(delta), flag


#tmp = sorted(rho.itervalues())


#for i in rho:
#    print i, 'rho', rho[i]

#print '# rho is density, delta is distance'

hd = ['#n', 'r']
print '\t'.join(map(str, hd))


#print nn

raise SystemExit()

#r = [delta[elem] * rho[elem] for elem in delta]
#r.sort(reverse=True)

#for i in xrange(len(r)):
#    j = [i, r[i]]
#    print '\t'.join(map(str, j))


# assign point to cluster







# select cluster
def tuning_point(r):
    n = len(r)
    r.sort(reverse=True)
    ts = []
    for i in xrange(n-1):
        k1i = r[i+1] - r[i]
        if i > 0:
            ki1 = (r[i] - r[0]) / i
        else:
            ki1 = 1
        t = k1i / ki1
        ts.append(t)

    n = len(ts)
    tmx = ts[0]
    m = 0
    for i in xrange(1, n):
        t = ts[i]
        if t > tmx:
            tmx = t
            m = i

    #print 'ts', ts[:10], ts[-10:], len(ts)
    return tmx, m

tmx, m = tuning_point(r)
print 'tuning point', tmx, m
import numpy as np
r = np.asarray(r)
r.sort()
r = r[::-1]

r0 = np.diff(r)
r1 = np.diff(r0)
print r
print r1, r1.max(), r1.min()





raise SystemExit()

hd = ['#name', 'density', 'dist']
print '\t'.join(map(str, hd))


rs = []
for i in delta:
    #print i, 'delta', delta[i]
    a, b = rho.get(i, 0), delta[i]
    r = a * b
    rs.append(r)
    j =  [i, a, b]
    print '\t'.join(map(str, j))



