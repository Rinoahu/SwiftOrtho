#!usr/bin/env python
# this script is used to cluster operons
# it need 2 files:
# 1. the operon files
# 2. the gene family file from orthomcl result

import sys
import networkx as nx

# the binary search algorithm
def bisect(xs, y, key=lambda x: x):
    N = len(xs)
    L, R = None, None
    # find most left
    l, r = 0, N - 1
    while r - l > 1:
        m = (l + r) // 2
        x = xs[m]
        if x < y:
            l = m + 1
        elif x > y:
            r = m
        else:
            break
    L = m

    # find most right
    l, r = 0, N - 1
    while r - l > 1:
        m = (l + r) // 2
        x = xs[m]
        if x < y:
            l = m
        elif x > y:
            r = m - 1
        else:
            break

    R = m
    return L, R


# parse the gene family and index
def gene_fam_idx0(f):
    groups = []
    flag = 0
    for i in f:
        j = i[:-1].split('\t')
        groups.extend([[elem, flag] for elem in j])
        flag += 1
    groups.sort()
    return groups


def operon_clust0(f, groups):

    # parse the operon
    operons = []
    for i in f:
        j = i[:-1].split('\t')[0]
        if j.startswith('gene_id'):
            continue
        # if '-->' in j:
        #   operon = j.split('-->')
        # else:
        #   operon = j.split('<--')
        # operons.append(operon)
        operons.append(j)

    # build operon index
    operondb = {}
    flag = 0
    for i in operons:
        for j in i:
            k = groups.get(j)
            if k:
                try:
                    operondb[k].append(flag)
                except:
                    operondb[k] = [flag]
        flag += 1

    # find operon and cluster
    for i in operons:
        idx = []
        for j in i:
            k = groups.get(j)
            if k:
                idx.extend(k)

        # query operon
        op0 = [groups.get(elem) for elem in i]
        idx = set(idx)
        for j in idx:
            op1 = [groups.get(elem) for elem in operons[j]]

    return operons, operondb


def gene_fam_idx(f):
    groups = {}
    flag = 0
    for i in f:
        j = i[:-1].split('\t')
        for k in j:
            groups[k] = flag
        flag += 1

    return groups


def operon_clust(f, groups):

    # parse the operon and build the operon query database
    operondb = {}
    operons = []
    flag = 0
    for i in f:
        op = i[:-1].split('\t')[0]
        if op.startswith('gene_id'):
            continue
        operons.append(op)
        ops = '-->' in op and op.split('-->') or op.split('<--')

        for j in ops:
            k = groups.get(j)
            if k:
                try:
                    operondb[k].append(flag)
                except:
                    operondb[k] = [flag]
        flag += 1

    # find operon and cluster
    N_ops = len(operons)
    G = nx.Graph()
    # for i in operons:
    for i0 in xrange(N_ops):
        i = operons[i0]

        sp0 = '-->' in i and '-->' or '<--'
        op0 = i.split(sp0)
        idxs = [operondb.get(groups[j], []) for j in op0 if j in groups]
        idxs = sum(idxs, [])
        idxs = set(idxs)

        # query operon
        group0 = [groups[elem] for elem in op0 if elem in groups]
        for j0 in idxs:
            j = operons[j0]
            sp1 = '-->' in j and '-->' or '<--'
            op1 = j.split(sp1)
            # group1 = [groups.get(elem) for elem in op1 if elem in groups]
            group1 = [groups[elem] for elem in op1 if elem in groups]
            share = set(group0).intersection(group1)
            N_shr = len(share) * 1.
            cv0 = N_shr / len(op0)
            cv1 = N_shr / len(op1)
            score = 2. * cv0 * cv1 / (cv0 + cv1)
            # score = N_shr
            if N_shr > 2 and max(cv1, cv0) > .5:
                print i, j, sp0.join(map(str, group0)), sp1.join(map(str, group1)), score
                G.add_edge(i0, j0, w=score)

    return G


def manual_print():
    print 'This script is used to cluster operons'
    print 'Usage:'
    print '  python this.py -g foo.gene.groups -p foo.operon'
    print 'Parameters:'
    print ' -g: protein/gene groups file. Each row contains the names of genes which belong to the same protein group/family.'
    print ' -p: operonic annotation file. The 1st column of this file should be like x0-->x1-->x2-->x3 or x0<--x1<--x2<--x3.'


if __name__ == '__main__':

    argv = sys.argv
	# recommand parameter:
    args = {'-g': '', '-p': ''}

    N = len(argv)
    for i in xrange(1, N):
        k = argv[i]
        if k in args:
    	    v = argv[i + 1]
            args[k] = v
        elif k[:2] in args and len(k) > 2:
    	    args[k[:2]] = k[2:]
        else:
            continue

    if args['-g'] == '' or args['-p'] == '':
        manual_print()
        raise SystemExit()

    try:
        genes, operons = args['-g'], args['-p']
    except:
        manual_print()
        raise SystemExit()

    # x = [1,1,1,1,1,2,2,2,3,3,3,3,4,4,4,4,4,5,5,5,5,5]
    # a, b = bisect(x, 2)
    # print a, b, 2, x[a], x[b]
    # if len(sys.argv[1:]) < 2:
    #    print 'python this.py foo.gene.groups foo.operon'
    #    raise SystemExit()
    #genes, operons = sys.argv[1:3]

    # build the gene family database
    f = open(genes, 'r')
    groups = gene_fam_idx(f)
    f.close()

    # print 'group number', len(groups)

    # build the operon database
    f = open(operons, 'r')
    G = operon_clust(f, groups)
    f.close()
    # print operondb
