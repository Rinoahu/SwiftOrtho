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

flag = 0
l2n = {}
f = open(qry, 'r')
_o = open(qry + '.abc', 'w')

for i in f:
    j = i[:-1].split('\t')
    o, x, y, z = j
    if x > y:
        continue

    if x not in l2n:
        l2n[x] = flag
        flag += 1

    if y not in l2n:
        l2n[y] = flag
        flag += 1

    X, Y = map(l2n.get, [x, y])
    out = '\t'.join(map(str, [X, Y, z]))
    _o.write(out + '\n')


f.close()
_o.close()


cmd = 'mcl %s.abc --abc -I 1.5 -o %s.abc.mcl -te %s'
os.system(cmd % (qry, qry, cpu))
