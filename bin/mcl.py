#!usr/bin/env python
import os
import sys
from mmap import mmap, ACCESS_WRITE, ACCESS_READ

# print the manual
def manual_print():
    print 'Usage:'
    print '    python mcl.py -i foo.xyz -I 1.5'
    print 'Parameters:'
    print '  -i: tab-delimited file which contain 3 columns'
    print '  -I: inflation parameter.'
    print '  -a: number of threads'
argv = sys.argv
# recommand parameter:
args = {'-i': '', '-I': '1.5', '-a': '1'}

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
    qry, ifl, cpu = args['-i'], float(args['-I']), int(args['-a']),

except:
    manual_print()
    raise SystemExit()


# correct the end of lines
def correct(s, m, l=None, r=None):
    if not l and not r:
        return s.rfind('\n', 0, m) + 1
    M = s.rfind('\n', l, m) + 1
    if l < M < r:
        return M
    else:
        M = s.find('\n', m, r) + 1
        return M

# bsearch for a sorted file  by given a query pattern
def binary_search(s, p, key=lambda x:x.split('\t', 1)[0], L = 0, R = -1):
    n = len(s)
    pn = len(p)
    R = R == -1 and n - 1 or R
    l = correct(s, L)
    r = correct(s, R)
    # find left
    while l < r:
        m = (l + r) // 2
        m = correct(s, m, l, r)
        if m == l or m == r:
            break
        t = s[m: s.find('\n', m)]
        pat = key(t)
        if pat >= p:
            r = m
        else:
            l = m

    # search from both direction
    left = m - 1
    while left >= 0:
        start = s.rfind('\n', 0, left)
        line = s[start+1: left]
        if key(line) == p:
            left = start
        else:
            break
    left += 1

    line = s[left: s.find('\n', left)]
    if key(line) != p:
        return -1, -1

    right = left
    while 1:
        end = s.find('\n', right)
        if key(s[right: end]) == p:
            right = end + 1
        else:
            break

    return left, right


# give a graph and a node, find neighbors of the node.
def neighbor(G, n):
    l, r = binary_search(G, n)
    lines = G[l:r].strip().split('\n')
    pairs = [n]
    for i in lines:
        if not i.strip():
            continue
        j = i.split('\t', 1)
        qid, sids = j[:2]
        if qid == n and sids:
            pairs.append(sids)
        else:
            continue
    out = '\t'.join(pairs)
    return out

# find connected componets in a graph
def connect(f, G):
    comps = set()
    pairs = {}
    for i in f:
        j = i[:-1].split('\t', 2)
        qid = j[0]
        # get new node and find new component
        if qid not in comps:
            if pairs:
                yield pairs

            stack = set([qid])
            visit = set()
            pairs = {}
            while stack:
                q = stack.pop()
                visit.add(q)
                pair = neighbor(G, q)
                t = [elem for elem in pair.split('\t')[1::2] if elem not in visit]
                pairs[q] = pair
                stack = stack.union(t)

            #print 'visit', len(visit), len(tmp), visit - tmp, qid, qid in (visit-tmp)
            #print 'share', visit.intersection(comps), len(visit.intersection(comps)), len(visit)
            comps = comps.union(visit)
        else:
            #print 'yes skip', len(comps)
            continue

    if pairs:
        yield pairs


# filter data
f = open(qry, 'r')
_o = open(qry+'.xyz', 'w')
for i in f:
    j = i[:-1].split('\t')
    #if j[1] > j[2]:
    #    j[1], j[2] = j[2], j[1]
    _o.write('\t'.join(j[1:4]) + '\n')
    j[1], j[2] = j[2], j[1]
    _o.write('\t'.join(j[1:4]) + '\n')

f.close()
_o.close()

# sort orthology results
os.system('export LC_ALL=C && sort -k1,2 --parallel=%s %s.xyz -o %s.xyz.srt' % (cpu, qry, qry))

# rewrite the file to single line
flag = None
out = []
_o = open('%s.xyz.srt.sg'%qry, 'w')
f = open('%s.xyz.srt'%qry, 'r')
for i in f:
    j = i[:-1].split('\t')
    #o, x, y, z = j[:4]
    x, y, z = j[:3]
    if flag != x:
        if out:
            output = '\t'.join(out)
            _o.write(output + '\n')
        flag = x
        out = [x, y, z]
    else:
        out.extend([y, z])

if out:
    output = '\t'.join(out)
    _o.write(output + '\n')

_o.close()
f.close()


f = open(qry + '.xyz.srt.sg', 'r')
G = mmap(f.fileno(), 0, access=ACCESS_READ)

flag = 0
for comp in connect(f, G):
    flag += len(comp)
    #print 'pair size', len(comp), flag
    for x in comp.values():
        y = x.split('\t')
        n = len(y)
        qid = y[0]
        for i in xrange(1, n, 2):
            w, z = y[i:i+2]
            if qid <= w:
                print '\t'.join([qid, w, z])

f.close()

#os.system('rm %s.ful %s.ful.sort'%(qry, qry))
