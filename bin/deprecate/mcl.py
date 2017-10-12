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


# correct the position
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
def binary_search0(s, p, key=lambda x:x.split('\t', 1)[0], L = 0, R = -1):
    mx = chr(255)
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
        #if pat[:pn] >= p:
        if pat+mx >= p+mx:
            r = m
        else:
            l = m

    # search from both direction
    left = r - 1
    while left >= 0:
        start = s.rfind('\n', 0, left)
        line = s[start+1: left]
        if key(line).startswith(p):
            left = start
        else:
            break
    left += 1

    line = s[left: s.find('\n', left)]
    if not key(line).startswith(p):
        return -1, -1

    right = left
    while 1:
        end = s.find('\n', right)
        if key(s[right: end]).startswith(p):
            right = end + 1
        else:
            break

    return left, right

def binary_search(s, p, key=lambda x:x.split('\t', 1)[0], L = 0, R = -1):
    #mx = chr(255)
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
        #if pat[:pn] >= p:
        #if pat+mx >= p+mx:
        if pat >= p:
            r = m
        else:
            l = m

    #print 'mid is', key(s[m: s.find('\n', m)]), p

    # search from both direction
    #left = r - 1
    left = m - 1
    while left >= 0:
        start = s.rfind('\n', 0, left)
        line = s[start+1: left]
        #if key(line).startswith(p):
        if key(line) == p:
            left = start
        else:
            break
    left += 1

    line = s[left: s.find('\n', left)]
    #if not key(line).startswith(p):
    if key(line) != p:
        return -1, -1

    right = left
    while 1:
        end = s.find('\n', right)
        #if key(s[right: end]).startswith(p):
        if key(s[right: end]) == p:
            right = end + 1
        else:
            break

    return left, right


# give a graph and a node, find neighbors of the node.
def neighbor0(G, n):
    l, r = binary_search(G, n)
    pat = [elem.split('\t') for elem in G[l:r].strip().split('\n')]
    #print 'query is', n, 'taget', pat
    if pat:
        return [elem[1] for elem in pat if elem[0] == n]
    else:
        return []


def neighbor(G, n):
    l, r = binary_search(G, n)
    lines = G[l:r].strip().split('\n')
    pat = [elem.split('\t') for elem in lines]
    sids = set()
    pairs = {}
    for i in lines:
        j = i.split('\t')
        qid, sid = j[:2]
        if qid == n:
            sids.add(sid)
            if qid < sid:
                try:
                    pairs[qid].append(i)
                except:
                    pairs[qid] = [i]
        else:
            continue

    return sids, pairs



# find connected componets in a graph
def connect(f, G):
    comps = set()
    comp = set()
    stack = set()
    pairs = {}
    for i in f:
        j = i[:-1].split('\t', 3)
        x, y = j[:2]
        # get new node and find new component
        if x not in comps:
            #if comp:
            if comp and pairs:
                #yield comp
                yield comp, pairs

            stack.clear()
            stack = set([x])
            #comp = set()
            comp.clear()
            #pairs = {}
            pairs.clear()
            while stack:
                q = stack.pop()
                comp.add(q)
                sids, pair = neighbor(G, q) 
                #t = [elem for elem in neighbor(G, q) if elem not in comp]
                t = [elem for elem in sids if elem not in comp]
                pairs.update(pair)
                stack = stack.union(t)

            comp -= comps
            comps = comps.union(comp)

        else:
            continue

    if comp:
        yield comp, pairs


# double and sort the file
f = open(qry, 'r')
_o = open(qry + '.ful', 'w')
for i in f:
    j = i[:-1].split('\t')
    x, y, z = j[1:4]
    out_0 = '\t'.join([x, y, z]) + '\n'
    out_1 = '\t'.join([y, x, z]) + '\n'
    _o.write(out_0)
    _o.write(out_1)

f.close()
_o.close()

#os.system('sort -k1 --parallel=%s %s.ful -o %s.ful.sort' % (cpu, qry, qry))
os.system('export LC_ALL=C && sort --parallel=%s %s.ful -o %s.ful.sort' % (cpu, qry, qry))


f = open(qry + '.ful.sort', 'r')
G = mmap(f.fileno(), 0, access=ACCESS_READ)

for i, j in connect(f, G):
    #i.split(ort()
    #i = sorted(i)
    #print '\t'.join(i)
    #print len(i), map(j.get, i)
    #print '\t'.join(map(j.get, i))
    for k in i:
        for x in j.get(k, []):
            print x

f.close()

#os.system('rm %s.ful %s.ful.sort'%(qry, qry))
