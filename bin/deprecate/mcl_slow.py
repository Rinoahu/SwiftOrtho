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
    pairs = [elem for elem in G[l:r].strip().split('\n') if elem]
    return pairs

# find connected componets in a graph
def connect(qry, G):
    comps = set()
    flag = 0
    for i in qry:
        j = i[:-1].split('\t', 3)
        x, y = j[:2]
        # get new node and find new component
        if x not in comps:
            stack = set([x])
            visit = set()
            while stack:
                q = stack.pop()
                visit.add(q)
                pairs = neighbor(G, q)
                for pair in pairs:
                    #print 'pair is', pair, len(pair)
                    x, y, z = pair.split('\t')[:3]
                    if y not in visit:
                        #stack = stack.union(y)
                        stack.add(y)
                        if x <= y:
                            #print pair
                            yield str(flag) + '\t' + pair

            comps = comps.union(visit)
            flag += 1

        else:
            continue


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

for i in connect(f, G):
    print i

f.close()

#os.system('rm %s.ful %s.ful.sort'%(qry, qry))
