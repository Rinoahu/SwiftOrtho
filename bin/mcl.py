#!usr/bin/env python
import os
import sys

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
    qry, ifl, cpu = args['-i'], float(args['-I']), int(args['-a']), 

except:
    manual_print()
    raise SystemExit()



# correct the position
correct = lambda s, i: s.rfind('\n', 0, i) + 1

# bsearch for a sorted file  by given a query pattern
def binary_search(s, p, L = 0, R = -1):
    n = len(s)
    pn = len(p)
    R = R == -1 and n - 1 or R
    l = correct(s, L)
    r = correct(s, R)
    # find left
    while l <= r:
        m = (l + r) // 2
        m = correct(s, m)
        if m == l:
            if s[m: end] >= p:
                r = m
            else:
                l = m
            break
        end = m + pn
        if s[m: end] >= p:
            r = m
        else:
            l = m

    left = r
    if s[left: left + pn] != p:
        return -1, -1
    right = r
    while 1:
        right = s.find('\n', right)
        if right != -1 and s[right + 1: right + 1 + pn] == p:
            right += 1
        else:
            break

    return left, right


# give a graph and a node, find neighbors of the node.
def neighbor(G, n):
	l, r = binary_search(G, n)
	pat = G[l:r]
	if pat:
		return pat.split('\n')
	else:
		return []


# find connected componets in a graph
def connect(f, G):
    comps = set()
    comp = set()
    for i in f:
        j = i[:-1].split('\t', 3)
        x, y = j[:2]
		# get new node and find new component
        if x not in comps:
            if comp:
                yield comp

            stack = set([x])
            comp = set()
	        while stack:
    	        q = stack.pop()
        	    comp.add(q)
            	t = [elem for elem neighbor(G, q) if elem not in comp]
	            stack = stack.union(t)
	        comps = comps.union(comp)

        else:
            continue

    if comp:
        yield comp




# get file
qry = sys.argv[1]
pat = sys.argv[2]

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

os.system('sort --parallel=%s %s.ful %s.ful.sort'%(cpu, qry, qry))


f = open(qry, 'r')
G = mmap(f.fileno(), 0, access = ACCESS_READ)

for i in connect(G, f):
    print '\t'.join(i)


f.close()





