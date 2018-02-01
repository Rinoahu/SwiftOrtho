#!usr/bin/env python
import os
import sys
from struct import pack, unpack
import numpy as np
from scipy import sparse
from collections import Counter
import networkx as nx
from itertools import izip
import gc

# print the manual
def manual_print():
    print 'Usage:'
    print '    python mcl.py -i foo.xyz -I 1.5'
    print 'Parameters:'
    print '  -i: tab-delimited file which contain 3 columns'
    print '  -I: inflation parameter for mcl'
    print '  -a: cpu number'


argv = sys.argv
# recommand parameter:
args = {'-i': '', '-I': '1.5', '-a': '4'}

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
    qry, ifl, cpu = args['-i'], float(args['-I']), int(args['-a'])

except:
    manual_print()
    raise SystemExit()


def normalize(x, norm='l1', axis=0):
    cs = x.sum(axis)
    y = np.asarray(cs)[0]
    if y.min() == 0 and y.max() > 0:
        y += y.nonzero()[0].min() / 1e3
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

        # normalization of col
        normalize(x, norm='l1', axis=0)
        if i % check == 0:
            x_old = x.copy()

        # expension
        x **= E

        # inflation
        x.data **= I

        # stop if no change
        if i % check == 0 and i > 0:
            if (abs(x - x_old) - rtol * abs(x_old)).max() <= atol:
                break

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


# mcl wrapper for x,y,z format
def mcl_xyz(f):
    l2n = {}
    dmx = 0
    # for x,y,z in xyz:
    for i in f:
        x, y = i.split('\t', 3)[:2]
        if x not in l2n:
            l2n[x] = dmx
            dmx += 1
        if y not in l2n:
            l2n[y] = dmx
            dmx += 1

    f.seek(0)
    dmx += 1
    G_d = sparse.lil_matrix((dmx, dmx), dtype='float32')
    # for x,y,z in xyz:
    for i in f:
        x, y, z = i.split('\t', 4)[:3]
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
        # print '\t'.join([n2l[elem] for elem in i])
        out = '\t'.join([n2l[elem] for elem in i])
        yield out


# connected based mcl
def cnc(qry, alg='mcl', rnd=2, chk=10**7):
    flag = 0
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
    while NNs:
        x, j = NNs.popitem()
        for y in j[1:]:
            G.add_edge(x, y)


    l2n = {}
    flag = 0
    for i in nx.connected_components(G):
        for j in i:
            l2n[j] = flag

        flag += 1

    del G
    gc.collect()

    # get 2nd round of cluster
    #G1 = nx.Graph()
    G1 = Counter()
    f = open(qry, 'r')
    for i in f:
        j = i[:-1].split('\t')
        if len(j) == 4:
            x, y, z = j[1:4]
        else:
            x, y, z = j[:3]

        if x > y:
            continue

        X, Y = map(l2n.get, [x, y])
        Z = float(z)
        if X and Y:
            key = X < Y and (X, Y) or (Y, X)
            G1[key] += Z
            #G1[key] += 1

    f.close()

    if rnd > 1:
        _o = open(qry+'.abc', 'w')
        for key in G1:
            x, y = key
            z = G1[key]
            #print x, y
            _o.write('%d\t%d\t%f\n'%(x, y, z))

        _o.close()
        os.system('mcl %s.abc --abc -q x -V all -I 1.2 -te %d -o %s.abc.mcl'%(qry, cpu, qry))

        G1 = []
        f = open(qry+'.abc.mcl', 'r')
        for pth in f:
            pths = pth[:-1].split('\t')
            G1.append(map(int, pths))
            #G1.add_path(pths)

        f.close()
        #os.system('rm %s.abc %s.abc.mcl'%(qry, qry))

    n2n = {}
    flag = 0
    for i in G1:
        for j in i:
            n2n[j] = flag

        flag += 1

    del G1
    gc.collect()

    print 'cluster number', flag, len(n2n)

    for i in l2n:
        j = l2n[i]
        l2n[i] = n2n.get(j, -1)

    # sort coo file and merge
    _o = open(qry + '.abcd', 'wb')
    f = open(qry, 'r')
    for i in f:
        j = i[:-1].split('\t')
        if len(j) == 4:
            x, y, z = j[1:4]
        else:
            x, y, z = j[:3]

        if x > y:
            continue

        cx, cy = map(l2n.get, [x, y])
        if cx and cy and cx == cy:
            out = '\t'.join(map(str, [cx, x, y, z]))
            _o.write(out + '\n')

    _o.close()

    os.system('export LC_ALL=C && sort -n --parallel=%d %s.abcd -o %s.abcd.srt && mv %s.abcd.srt %s.abcd' %
              (cpu, qry, qry, qry, qry))
    #raise SystemExit()

    os.system('rm -f %s.mcl' % (qry))
    cls = None
    flag = 0
    #_oopt = open(qry+'.mcl', 'w')
    _o = open(qry + '.abc', 'wb')
    f = open(qry + '.abcd', 'r')
    for i in f:
        c = i.split('\t', 2)[0]
        if c != cls:
            #if flag > 0 and flag % chk == 0:
            if flag > chk:
                _o.close()
                os.system('mcl %s.abc --abc -q x -V all -I %f -te %d -o %s.mcl_mem && cat %s.mcl_mem >> %s.mcl'%(qry, ifl, cpu, qry, qry, qry))
                #f1 = open(qry + '.abc', 'r')
                #for cl in mcl_xyz(f1):
                #    #print cl
                #    _oopt.write(cl+'\n')
                #f1.close()
                _o = open(qry + '.abc', 'wb')
                flag = 0

            cls = c

        j = i.split('\t')
        k = '\t'.join(j[1:])
        _o.write(k)
        flag += 1

    _o.close()
    os.system('mcl %s.abc --abc -q x -V all -I %f -te %d -o %s.mcl_mem &&  cat %s.mcl_mem >> %s.mcl'%(qry, ifl, cpu, qry, qry, qry))
    #f1 = open(qry + '.abc', 'r')
    #mcl_xyz(f1)
    #for cl in mcl_xyz(f1):
    #    #print cl
    #    _oopt.write(cl+'\n')
    #f1.close()
    #_oopt.close()

    #os.system('rm -f %s.abc %s.abcd %s.mcl_mem'%(qry, qry, qry))
    f.close()

if __name__ == '__main__':
    cnc(qry)
#cnc(qry)
