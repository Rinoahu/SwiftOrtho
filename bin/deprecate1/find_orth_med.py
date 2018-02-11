#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#  Copyright © xh
# CreateTime: 2016-06-03 13:55:13

# this is the py version of blast
import sys
import networkx as nx
from math import log10
import os
from commands import getoutput
from mmap import mmap, ACCESS_WRITE, ACCESS_READ
from collections import Counter


# print the manual
def manual_print():
    print 'Usage:'
    print '    python fast_search.py -i foo.sc [-c .5] [-y 50] [-n no]'
    print 'Parameters:'
    print '  -i: tab-delimited file which contain 14 columns'
    print '  -c: min coverage of sequence [0~1]'
    print '  -y: identity [0~100]'
    print '  -n: normalization score [no|bsr|bal]. bsr: bit sore ratio; bal:  bit score over anchored length. Default: no'
    print '  -a: cpu number for sorting. Default: 1'
    print '  -t: keep tmpdir[y|n]. Default: n'


argv = sys.argv
# recommand parameter:
args = {'-i':'', '-c':.5, '-y':0, '-n':'no', '-t':'n', '-a':'4'}

N = len(argv)
for i in xrange(1, N):
    k = argv[i]
    if k in args:
        try:
            v = argv[i+1]
        except:
            break
        args[k] = v
    elif k[:2] in args and len(k) > 2:
        args[k[:2]] = k[2:]
    else:
        continue

if args['-i']=='':
    manual_print()
    raise SystemExit()

try:
    qry, coverage, identity, norm, tmpdir, np = args['-i'], float(args['-c']), float(args['-y']), args['-n'], args['-t'], int(args['-a'])
except:
    manual_print()
    raise SystemExit()


#qry = sys.argv[1]
qry = os.path.abspath(qry)
fn = qry.split(os.sep)[-1]
os.system('mkdir -p %s_tmp/'%qry)
os.system('ln -sf %s %s_tmp/'%(qry, qry))
qry = qry + '_tmp/' + fn

# blast parser, return list contains blast results with the same query id
# remove the duplicated pairs or qid-sid
def blastparse(f, coverage = .5, identity = 0., norm='no', len_dict = {}):
    output = {}
    #len_dict = {}
    flag = None
    # max bit score
    mbsc = -1
    for i in f:
        j = i[: -1].split('\t')
        qid, sid = j[:2]
        qtx, stx = qid.split('|')[0], sid.split('|')[0]
        key = sid
        idy, aln, mis, gop, qst, qed, sst, sed, evalue, score = map(float, j[2:12])
        # the fastclust seq search format
        if len(j) > 13:
            qln, sln = map(float, j[12:14])
        else:
            if qid in len_dict:
                qln = len_dict[qid]
            else:
                qln = max(qst, qed)
                len_dict[qid] = qln
            if sid in len_dict:
                sln = len_dict[sid]
            else:
                sln = max(sst, sed)
                len_dict[sid] = sln


        qcv = (1.+abs(qed-qst))/qln
        scv = (1.+abs(sed-sst))/sln
        if qcv<coverage or scv<coverage or idy<identity:
            continue

        if flag != qid:
            if output:
                yield output.values()

            mbsc = score
            #print 'max bit score is', mbsc, qid, sid
            output = {}
            length = aln
            flag = qid
            if norm == 'bsr':
                Score = score / mbsc
            elif norm == 'bal':
                Score = score / aln
            else:
                Score = score
            output[key] = [qid, sid, Score]
        else:

            if norm == 'bsr':
                Score = score / mbsc
            elif norm == 'bal':
                Score = score / aln
            else:
                Score = score

            if key not in output or output[key][-1]<Score:
                output[key] = [qid, sid, Score]

    if output:
        yield output.values()


# distinguish IP and O
# return the IP and O
def get_IPO(hits, l2n = {}):
    # get max of each species
    sco_max = Counter()
    out_max = 0
    for hit in hits:
        qid, sid, sco = hit
        sco = float(sco)
        qtx = qid.split('|')[0]
        stx = sid.split('|')[0]
        sco_max[stx] = max(sco_max[stx], sco)
        if qtx != stx:
            out_max = max(out_max, sco)

    visit = set()
    ips, ots, cos  = [], [], []
    for hit in hits:
        qid, sid, sco = hit
        if sid not in l2n:
            continue
        x, y = map(l2n.get, [qid, sid])
        sco = float(sco)
        if sid in visit:
            continue
        else:
            visit.add(sid)
        qtx = qid.split('|')[0]
        stx = sid.split('|')[0]
        #out = [qid, sid, sco]
        out = [x, y, sco]
        if qtx == stx:
            if sco >= out_max:
                #ips.append(hit)
                ips.append(out)
            else:
                continue
        else:
            if sco >= sco_max[stx]:
                #ots.append(hit)
                ots.append(out)
            else:
                cos.append(out)
    ips.sort()
    ots.sort()
    cos.sort()
    IPs = ['\t'.join(map(str, elem)) + '\n' for elem in ips]
    OTs = ['\t'.join(map(str, elem)) + '\n' for elem in ots]
    COs = ['\t'.join(map(str, elem)) + '\n' for elem in cos]
    return IPs, OTs, COs

# parse and find IP, O from blast results
# locus 2 number
l2n = {}
n2l = []
flag = 0
len_dict = {}
f = open(qry, 'r')
for i in f:
    j = i.split('\t')
    #if len(j) > 13:
    #    break
    qid, sid = j[:2]
    idy, aln, mis, gop, qst, qed, sst, sed, evalue, score = map(float, j[2:12])
    len_dict[qid] = max(qst, qed, len_dict.get(qid, 0))
    if qid not in l2n:
        l2n[qid] = flag
        flag += 1
        n2l.append(qid)
   
f.close()

f = open(qry, 'r')
qip = qry + '.qIPs.txt'
_oqips = open(qip, 'w')

qot = qry + '.qOTs.txt'
_oqots = open(qot, 'w')

qco = qry + '.qCOs.txt'
_oqcos = open(qco, 'w')

for i in blastparse(f, coverage, identity, norm, len_dict):
    IPs, OTs, COs = get_IPO(i, l2n)
    #print IPs, OTs, COs, l2n
    _oqips.writelines(IPs)
    _oqots.writelines(OTs)
    _oqcos.writelines(COs)

_oqips.close()
_oqots.close()
_oqcos.close()

# correct search results
def correct(s, m, l=None, r=None, sep='\n'):
    if not l and not r:
        return s.rfind(sep, 0, m) + 1
    M = s.rfind(sep, l, m) + 1
    if l < M < r:
        return M
    else:
        M = s.find(sep, m, r) + 1
        return M

def binary_search(s, p, key=lambda x:x.split('\t', 1)[0], L=0, R=-1, sep='\n'):
    #mx = chr(255)
    n = len(s)
    #pn = len(p)
    R = R == -1 and n-1 or R
    l = correct(s, L, sep=sep)
    r = correct(s, R, sep=sep)
    # find left
    while l < r:
        m = (l + r) // 2
        m = correct(s, m, l, r, sep=sep)
        if m == l or m == r:
            break
        t = s[m: s.find(sep, m)]
        pat = key(t)
        if pat >= p:
            r = m
        else:
            l = m

    left = m - 1
    while left >= 0:
        start = s.rfind(sep, 0, left)
        line = s[start+1: left]
        if key(line) == p:
            left = start
        else:
            break
    left += 1

    line = s[left: s.find(sep, left)]
    if key(line) != p:
        return -1, -1, []

    right = left
    while 1:
        end = s.find(sep, right)
        try:
            target = key(s[right: end])
        except:
            target = None 
        #if key(s[right: end]) == p:
        if target == p:
            right = end + 1
        else:
            break

    pairs = s[left: right].strip().split(sep)
    return left, right, pairs


###############################################################################
# get OTs
###############################################################################
inots = [-1] * len(l2n)
ots = qry + '.OTs.txt'
_oots = open(ots, 'w')
f = open(qot, 'r')
S = mmap(f.fileno(), 0, access = ACCESS_READ)
for i in f:
    qid, sid, qsc = i.split('\t')
    qid, sid = map(int, [qid, sid])
    if qid < sid:
        st, ed, pairs = binary_search(S, [sid, qid], lambda x: map(int, x.split('\t', 3)[:2]))
        if pairs:
            x, y, sco = pairs[0].split('\t')
            nsc = (float(qsc) + float(sco)) / 2.
            _oots.write('%d\t%d\t%f\n'%(qid, sid, nsc))
            inots[qid] = 1
            inots[sid] = 1

_oots.close()
f.close()


###############################################################################
# get IPs
###############################################################################
ipqa = {}
IPqA = {}
ips = qry + '.IPs.txt'
_oips = open(ips, 'w')
ipss = qry + '.IPss.txt'
_oipss = open(ipss, 'w')

f = open(qip, 'r')
try:
    S = mmap(f.fileno(), 0, access = ACCESS_READ)
except:
    S = ''

flag = None
output = []
for i in f:
    qid, sid, qsc = i.split('\t')
    qid, sid = map(int, [qid, sid])
    st, ed, pairs = binary_search(S, [sid, qid], lambda x: map(int, x.split('\t', 3)[:2]))
    if not pairs:
        continue
    if qid < sid:
        x, y, sco = pairs[0].split('\t')
        nsc = (float(qsc) + float(sco)) / 2.
        _oips.write('%d\t%d\t%f\n'%(qid, sid, nsc))

        if inots[qid] == 1 or inots[sid] == 0:
            qtx = n2l[qid].split('|')[0]
            try:
                ipqa[qtx][0] += nsc
                ipqa[qtx][1] += 1.
            except:
                ipqa[qtx] = [nsc, 1.]

        qtx = n2l[qid].split('|')[0]
        try:
            IPqA[qtx][0] += nsc
            IPqA[qtx][1] += 1.
        except:
            IPqA[qtx] = [nsc, 1.]

    if qid != flag:
        if output:
            output = '\t'.join(map(str, output)) + '\n'
            _oipss.write(output)
        flag = qid
        output = [qid, sid]
        #print 'output is', output
    else:
        output.append(sid)

if output:
    output = '\t'.join(map(str, output)) + '\n'
    _oipss.write(output)

for k in IPqA:
    a, b = k in ipqa and ipqa[k] or IPqA[k]
    IPqA[k] = a / b

#print 'IPqA', IPqA

_oips.close()
_oipss.close()
f.close()


###############################################################################
# get COs
###############################################################################
cos = qry + '.COs.txt'
_ocos = open(cos, 'w')

fipss = open(ipss, 'r')
Sipss = mmap(fipss.fileno(), 0, access = ACCESS_READ)

fqco = open(qco, 'r')
Sqco = mmap(fqco.fileno(), 0, access = ACCESS_READ)

f = open(ots, 'r')
for i in f:
    # get ot pair
    qid, sid, sco = i.split('\t')[:3]
    qid, sid = map(int, [qid, sid])
    # get ip of ot
    st, ed, pairs = binary_search(Sipss, qid, lambda x: int(x.split('\t', 2)[0]))
    if not pairs:
        continue
    qips = map(int, pairs[0].split('\t'))
    st, ed, pairs = binary_search(Sipss, sid, lambda x: int(x.split('\t', 2)[0]))
    if not pairs:
        continue

    visit = set()
    sips = map(int, pairs[0].split('\t'))
    for qip in qips:
        for sip in sips:
            if qip != qid or sip != sid:
                if (qip, sip) not in visit and qip < sip:
                    visit.add((qip, sip))
                else:
                    continue

                st, ed, pairs = binary_search(Sqco, [qip, sip], lambda x: map(int, x.split('\t', 3)[:2]))
                if not pairs:
                    st, ed, pairs = binary_search(Sqco, [sip, qip], lambda x: map(int, x.split('\t', 3)[:2]))
                if pairs:
                    x, y, sco = pairs[0].split('\t')
                    if int(x) > int(y):
                        x, y = y, x
                    _ocos.write('\t'.join([x, y, sco]) + '\n')
                    #_ocos.write(pairs[0]+'\n')
                else:
                    continue
                   
_ocos.close()
f.close()

###############################################################################
# print normalized IPs
###############################################################################
f = open(ips, 'r')
for i in f:
    x, y, score = i[:-1].split('\t')
    x, y = map(int, [x, y])
    qid, sid = n2l[x], n2l[y]
    tax = qid.split('|')[0]
    avg = IPqA[tax]
    score = float(score)
    try:
        out = map(str, ['IP', qid, sid, score/avg])
    except:
        continue
    print '\t'.join(out)

f.close()
IPqA.clear()


# get co or ot from same taxon
def get_sam_tax(f, n2l):
    flag = None
    out = []
    visit = set()
    for i in f:
        x, y, sco = i[:-1].split('\t')
        x, y = map(int, [x, y])

        if (x, y) not in visit:
            visit.add((x, y))
        else:
            continue

        qid, sid = n2l[x], n2l[y]
        qtx = qid.split('|')[0]
        sco = float(sco)
        if qtx != flag:
            if out:
                yield out
            flag = qtx
            out = [[qid, sid, sco]]
        else:
            out.append([qid, sid, sco])
    if out:
        yield out

# normal co or ot
def n_co_ot(out):
    avgs = {}
    for qid, sid, sco in out:
        stx = sid.split('|')[0]
        try:
            avgs[stx][0] += sco
            avgs[stx][1] += 1.
        except:
            avgs[stx] = [sco, 1.]
    for k in avgs:
        a, b = avgs[k]
        avgs[k] = a / b

    for qid, sid, sco in out:
        stx = sid.split('|')[0]
        avg = avgs[stx]
        yield [qid, sid, sco / avg]


###############################################################################
# print normalized OTs and COs
###############################################################################
f = open(ots, 'r')
for i in get_sam_tax(f, n2l):
    for j in n_co_ot(i):
        out = '\t'.join(map(str, j))
        print 'OT\t' + out

f.close()

f = open(cos, 'r')
for i in get_sam_tax(f, n2l):
    for j in n_co_ot(i):
        out = '\t'.join(map(str, j))
        print 'CO\t' + out

f.close()
