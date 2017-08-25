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
args = {'-i':'', '-c':.5, '-y':50, '-n':'no', '-t':'n', '-a':'4'}

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
    Qry, coverage, identity, norm, tmpdir, np = args['-i'], float(args['-c']), float(args['-y']), args['-n'], args['-t'], int(args['-a'])
except:
    manual_print()
    raise SystemExit()

#print 'norm is', norm
# filter the m8 file by coverage and identity
qry = Qry + '.flt'
_o = open(qry, 'w')
f = open(Qry, 'r')
for i in f:
    j = i.split('\t')
    idy, aln, mis, gop, qst, qed, sst, sed, evalue, score = map(float, j[2:12])
    # the fastclust seq search format
    if len(j) > 13:
        qln, sln = map(float, j[12:14])
    else:
        qln, sln = max(qst, qed), max(sst, sed)


    qcv = (1.+abs(qed-qst))/qln
    scv = (1.+abs(sed-sst))/sln
    if qcv < coverage or scv < coverage or idy < identity:
        continue
    else:
        _o.write(i)

f.close()
_o.close()

#qry = sys.argv[1]
qry = os.path.abspath(qry)
na = qry.split(os.sep)[-1]
#print 'make dir', qry
os.system('mkdir -p %s_tmp/'%qry)
os.system('ln -sf %s %s_tmp/'%(qry, qry))
Qry=qry
#os.system('ln -sf %s %s.tmp'%(qry, qry))
qry = qry + '_tmp/' + na

#try:
#    identity = eval(sys.argv[2])
#except:
#    identity = 50
#try:
#    coverage = eval(sys.argv[3])
#except:
#    coverage = .5

# blast parser, return list contains blast results with the same query id
# remove the duplicated pairs or qid-sid
def blastparse(f, coverage = .5, identity = 50, norm='no', len_dict = {}):
    output = {}
    #len_dict = {}
    flag = None
    # max bit score
    mbsc = -1
    for i in f:
        j = i[: -1].split('\t')
        qid, sid = j[:2]
        qtx, stx = qid.split('|')[0], sid.split('|')[0]
        key = qtx == stx and sid or stx

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


def blastparse0(f, coverage = .5, identity = 50, norm='no'):
    output = {}
    flag = None
    # max bit score
    mbsc = -1
    for i in f:
        j = i[: -1].split('\t')
        qid, sid = j[:2]
        qtx, stx = qid.split('|')[0], sid.split('|')[0]
        key = qtx == stx and sid or stx

        idy, aln, mis, gop, qst, qed, sst, sed, evalue, score = map(float, j[2:12])
        # the fastclust seq search format
        if len(j) > 13:
            qln, sln = map(float, j[12:14])
        else:
            qln, sln = max(qst, qed), max(sst, sed)

        qcv = (1.+abs(qed-qst))/qln
        scv = (1.+abs(sed-sst))/sln
        #mbsc = score > mbsc and score or mbsc
        #mbsc = mbsc < 0 and score or mbsc
        #print 'max bit score is', mbsc, qid, sid

        if flag != qid:
            if output:
                yield output.values()

            mbsc = score
            #print 'max bit score is', mbsc, qid, sid
            output = {}
            length = aln
            flag = qid
            if qcv>=coverage and scv>=coverage and idy>=identity:
                if norm == 'bsr':
                    Score = score / mbsc
                elif norm == 'bal':
                    Score = score / aln
                else:
                    Score = score

                output[key] = [qid, sid, Score]
        else:
            if qcv>=coverage and scv>=coverage and idy>=identity:

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
def get_IPO(output_flt):
    output_flt.sort(key = lambda x: [x[-1], x[0].split('|')[0] == x[1].split('|')[0]], reverse = True)
    #size = output_flt[0][3]
    IPs, Os = [], []
    flag = 'IP'
    for i in output_flt:
        qid, sid, Score = i
        # score can be changed
        qtx, stx = qid.split('|')[0], sid.split('|')[0]
        if qtx == stx and flag == 'IP':
            out = qid < sid and [qid, sid, Score] or [sid, qid, Score]
            out = '\t'.join(map(str, out)) + '\n'
            IPs.append(out)
        elif qtx != stx: 
            out = qid < sid and [qid, sid, Score] or [sid, qid, Score]
            out = '\t'.join(map(str, out)) + '\n'
            Os.append(out)
            flag = 'O'
        else:
            continue

    #print 'IPs length', len(IPs)
    return IPs, Os


def get_IPO0(output_flt):
    output_flt.sort(key = lambda x: [x[-1], x[0].split('|')[0] == x[1].split('|')[0]], reverse = True)
    #size = output_flt[0][3]
    visit = set()
    IPs, Os = [], []
    flag = 'IP'
    for i in output_flt:
        qid, sid, Score = i
        # score can be changed
        qtx, stx = qid.split('|')[0], sid.split('|')[0]
        if qtx == stx and flag == 'IP':
            out = qid < sid and [qid, sid, Score] or [sid, qid, Score]
            out = '\t'.join(map(str, out)) + '\n'
            IPs.append(out)
        elif qtx != stx: 
            flag = 'O'
            if (qtx, stx) not in visit:
                out = qid < sid and [qid, sid, Score] or [sid, qid, Score]
                out = '\t'.join(map(str, out)) + '\n'
                Os.append(out)
            visit.add((qtx, stx))
        else:
            continue

    #print 'IPs length', len(IPs)
    return IPs, Os


# parse and find IP, O from blast results
len_dict = {}
f = open(qry, 'r')
for i in f:
    j = i.split('\t')
    if len(j) > 13:
        break
    qid, sid = j[:2]
    idy, aln, mis, gop, qst, qed, sst, sed, evalue, score = map(float, j[2:12])
    len_dict[qid] = max(qst, qed, len_dict.get(qid, 0))
    len_dict[sid] = max(sst, sed, len_dict.get(sid, 0))
   

f.close()

f = open(qry, 'r')
qip = qry + '.qIPs.txt'
_oIPs = open(qip, 'w')

qo = qry + '.qOs.txt'
_oOs = open(qo, 'w')
for i in blastparse(f, coverage, identity, norm, len_dict):
    IPs, Os = get_IPO(i)
    #print IPs, Os
    _oIPs.writelines(IPs)
    _oOs.writelines(Os)

_oIPs.close()
_oOs.close()


# sort QIP and QO
qipsort = qip + '.sort'
qosort = qo + '.sort'

#os.system('sort --parallel=%s -k1,2 %s > %s;mv %s %s'%(np, qip, qipsort, qipsort, qip))
os.system('sort --parallel=%s -k1,2 %s -o %s;mv %s %s'%(np, qip, qipsort, qipsort, qip))

#os.system('sort --parallel=%s -k1,2 %s > %s;mv %s %s'%(np, qo, qosort, qosort, qo))
os.system('sort --parallel=%s -k1,2 %s -o %s;mv %s %s'%(np, qo, qosort, qosort, qo))

# get IPs and Os
def find_IPO(f):
    flag = None
    output = []
    for i in f:
        j = i[:-1].split('\t')
        qid, sid, score = j
        if flag != j[:2]:
            if len(output) == 4:
                yield output[0], output[1], sum(output[2:]) / 2.
            flag = j[:2]
            output = [qid, sid, float(score)]
        else:
            output.append(float(score))

    if len(output) == 4:
        yield output[0], output[1], sum(output[2:]) / 2.

# get IP
ipn = qry + '.IPs.txt'
_o = open(ipn, 'w')
f = open(qip, 'r')
for qid, sid, score in find_IPO(f):
    _o.write('%s\t%s\t%f\n'%(qid, sid, score))

_o.close()

# get O
osn = qry + '.Os.txt'
_o = open(osn, 'w')
f = open(qo, 'r')
for qid, sid, score in find_IPO(f):
    _o.write('%s\t%s\t%f\n'%(qid, sid, score))

_o.close()

# get CO
# correct the position
# binary search by lines
correct = lambda s, i: s.rfind('\n', 0, i) + 1
def binary_search(s, p, L = 0, R = -1):
    n = len(s)
    pn = len(p)
    R = R == -1 and n - 1 or R
    l = correct(s, L)
    r = correct(s, R)
    # find left
    end = pn
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
        pairs = s[-1: -1].split('\n')
        return -1, -1, pairs

    right = r
    while 1:
        right = s.find('\n', right)
        if right != -1 and s[right + 1: right + 1 + pn] == p:
            right += 1
        else:
            break

    pairs = s[left: right].split('\n')
    return left, right, pairs

# sort the IPs by k2
os.system("awk '{print $2\"\\t\"$1\"\\t\"$3}' %s > %s.tmp; sort --parallel=%s -k1 %s.tmp -o %s.k2; rm %s.tmp"%(ipn, ipn, np, ipn, ipn, ipn))
ipnk2 = ipn + '.k2'
f0k2 = open(ipnk2, 'r')
S0k2 = mmap(f0k2.fileno(), 0, access = ACCESS_READ)


f0 = open(ipn, 'r')
S0 = mmap(f0.fileno(), 0, access = ACCESS_READ)

f1 = open(osn, 'r')
S1 = mmap(f1.fileno(), 0, access = ACCESS_READ)

cosn = qry + '.COs.txt'
_o = open(cosn, 'w')
#f2 = open(osn, 'r')
#for i in f2:
for i in f1:
    qo, so = i.split('\t')[:2]
    #print 'qO', qo, 'sO', so
    #qip = set()
    qip = set([qo])
    #print 'binary search', binary_search(S0, qo)
    L0, R0, pairs0 = binary_search(S0, qo)
    L1, R1, pairs1 = binary_search(S0k2, qo)
    #pairs = S0[L: R].split('\n')
    #print 'pairs', pairs
    for j in pairs0 + pairs1:
        if j:
            a, b = j.split('\t')[: 2]
            qip.add(a)
            qip.add(b)

    #sip = set()
    sip = set([so])
    L0, R0, pairs0 = binary_search(S0, so)
    L1, R1, pairs1 = binary_search(S0k2, so)
    #pairs = S0k2[L: R].split('\n')
    #print pairs, pairsk2
    for j in pairs0 + pairs1:
        if j:
            a, b = j.split('\t')[: 2]
            sip.add(a)
            sip.add(b)

    #print 'qip, sip', qo, qip, so, sip
    # find the CO
    for i in qip:
        for j in sip:
            #print 'finding CO', qip, sip
            if (i==qo and j==so) or (i==so and j==qo) or (i==j):
                continue
            key = i < j and i + '\t' + j or j + '\t' + i
            L, R, pairs = binary_search(S1, key)
            if not [elem for elem in pairs if elem]:
                #print 'CO', i, j, S1[L: R]
                out = '\t'.join([i, j, S1[L:R]])
                _o.write(out+'\n')
            else:
                #print 'O', i, j, S1[L: R]
                continue

_o.close()

f0k2.close()
f0.close()
f1.close()
#f2.close()

# normalization
# IP
ipqa = Counter()
ipqa_n = Counter()
ipqA = Counter()
ipqA_n = Counter()
f = open(ipn, 'r')

for i in f:
    qid, sid, score = i[:-1].split('\t')
    score = float(score)
    tax = qid.split('|')[0]
    ipqA[tax] += score
    ipqA_n[tax] += 1.
    qL, qR, qpairs = binary_search(S1, qid)
    sL, sR, spairs = binary_search(S1, sid)
    if qL!=-1 and sL!=-1 and qR!=-1 and sR!=-1:
        ipqa[tax] += score
        ipqa_n[tax] += 1

f.close()


for i in ipqA:
    n = ipqa_n[i]
    if n > 0:
        ipqa[i] /= n
    else:
        ipqa[i] = ipqA[i] / ipqA_n[i]

del ipqA, ipqA_n, ipqa_n


f = open(ipn, 'r')
for i in f:
    qid, sid, score = i[:-1].split('\t')
    tax = qid.split('|')[0]
    n = ipqa[tax]
    score = float(score)
    #out = map(str, ['IP', qid, sid, score/n])
    try:
        out = map(str, ['IP', qid, sid, score/n])
    except:
        #print 'score is', tax, tax in ipqa, out
        continue
    print '\t'.join(out)

f.close()
del ipqa



# co and o
# sort blast with taxon name
#os.system("awk '{print $1\"\\t\"$2\"\\t\"$12}' %s | sort --parallel=4 > %s.srt"%(qry, qry))
#os.system("awk '{print $1\"\\t\"$2\"\\t\"$12}' %s > %s.xyz && sort --parallel=4 %s.xyz > %s.srt && rm %s.xyz"%(qry, qry, qry, qry, qry))
#os.system("awk '{if(NF>=14&&$3>=%f&&($8-$7)>=%f&&($10-$9)/$14>=%f)print $1\"\\t\"$2\"\\t\"$12}' %s > %s.xyz && sort --parallel=4 %s.xyz > %s.srt && rm %s.xyz"%(identity, coverage, coverage, qry, qry, qry, qry, qry))

#os.system("awk '{print $1\"\\t\"$2\"\\t\"$12}' %s > %s.xyz && sort --parallel=4 %s.xyz > %s.srt && rm %s.xyz"%(qry, qry, qry, qry, qry))
os.system("awk '{print $1\"\\t\"$2\"\\t\"$12}' %s > %s.xyz && sort --parallel=%s %s.xyz -o %s.srt && rm %s.xyz"%(qry, qry, np, qry, qry, qry))



dbs = open(qry+'.srt', 'r')
Dbs = mmap(dbs.fileno(), 0, access = ACCESS_READ)

# normal co and o
def normal_co_o(f):
    flag = None
    out = []
    for i in f:
        qtax, stax, typ, qid, sid, score = i[:-1].split('\t')
        score = float(score)
        if (qtax, stax) != flag:
            if out:
                yield out
            flag = (qtax, stax)
            out = [[qtax, stax, typ, qid, sid, score]]
        else:
            out.append([qtax, stax, typ, qid, sid, score])
    if out:
        yield out

# the co
_o = open(cosn+'.tmp', 'w')
f = open(cosn, 'r')
for i in f:
    #qid, sid, score = i[:-1].split('\t')
    qid, sid = i[:-1].split('\t')[:2]
    qL, qR, qpairs = binary_search(Dbs, qid+'\t'+sid)
    sL, sR, spairs = binary_search(Dbs, sid+'\t'+qid)
    if qL==-1 or qR==-1 or sL==-1 or sR==-1:
        continue
    qid, sid, qs = qpairs[0].split('\t')
    sid, qid, ss = spairs[0].split('\t')
    score = (float(qs) + float(ss)) / 2
    qtax = qid.split('|')[0]
    stax = sid.split('|')[0]
    out = map(str, [qtax, stax, 'CO', qid, sid, score])
    _o.write('\t'.join(out)+'\n')

f.close()
_o.close()

#os.system('sort --parallel=%s -k1,2 %s.tmp | uniq > %s.tmp.srt'%(np, cosn, cosn))
os.system('sort --parallel=%s -k1,2 %s.tmp -o %s.tmp.srt.tmp && rm %s.tmp && uniq %s.tmp.srt.tmp > %s.tmp.srt'%(np, cosn, cosn, cosn, cosn, cosn))


f = open(cosn+'.tmp.srt', 'r')
for i in normal_co_o(f):
    avg = sum([elem[-1] for elem in i]) * 1. / len(i)
    for j in i:
        qtax, stax, typ, qid, sid, score = j
        score = float(score)/avg
        typ = qid == sid and 'IP' or typ
        print '\t'.join(map(str, [typ, qid, sid, score]))

f.close()



# the os
_o = open(osn+'.tmp', 'w')
f = open(osn, 'r')
for i in f:
    qid, sid, score = i[:-1].split('\t')[:3]
    qtax = qid.split('|')[0]
    stax = sid.split('|')[0]
    out = map(str, [qtax, stax, 'O', qid, sid, score])
    _o.write('\t'.join(out)+'\n')

f.close()
_o.close()

#os.system('sort --parallel=%s -k1,2 %s.tmp | uniq > %s.tmp.srt'%(np, osn, osn))
os.system('sort --parallel=%s -k1,2 %s.tmp -o %s.tmp.srt.tmp && rm %s.tmp && uniq %s.tmp.srt.tmp > %s.tmp.srt'%(np, osn, osn, osn, osn, osn))



f = open(osn+'.tmp.srt', 'r')
for i in normal_co_o(f):
    avg = sum([elem[-1] for elem in i]) * 1. / len(i)
    for j in i:
        qtax, stax, typ, qid, sid, score = j
        score = float(score)/avg
        typ = qid == sid and 'IP' or typ
        #print '\t'.join([typ, qid, sid, score])
        print '\t'.join(map(str, [typ, qid, sid, score]))


f.close()

if tmpdir == 'n':
    os.system('rm -rf %s_tmp/'%Qry)

