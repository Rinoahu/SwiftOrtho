#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#  Copyright © xh
# CreateTime: 2016-06-03 13:55:13

# this is the py version of blast
import sys
#import networkx as nx
from math import log10
import os

#from subprocess import getoutput
if sys.version_info.major == 2:
    from commands import getoutput
else:
    from subprocess import getoutput

from mmap import mmap, ACCESS_WRITE, ACCESS_READ
from collections import Counter
import io

#open = io.open


# print the manual
def manual_print():
    print('Usage:')
    # print '    python find_orth.py -i foo.sc [-c .5] [-y 50] [-n no]'
    print('    python %s -i foo.sc [-c .5] [-y 50] [-n no]' % sys.argv[0])
    print('Parameters:')
    print('  -i: tab-delimited file which contain 14 columns')
    print('  -c: min coverage of sequence [0~1]')
    print('  -y: identity [0~100]')
    print(
        '  -n: normalization score [no|bsr|bal]. bsr: bit sore ratio; bal:  bit score over anchored length. Default: no')
    print('  -a: cpu number for sorting. Default: 1')
    print('  -t: keep tmpdir[y|n]. Default: n')
    print('  -T: tmpdir for sort command. Default: ./tmp/')


argv = sys.argv
# recommand parameter:
args = {'-i': '', '-c': .5, '-y': 0, '-n': 'no',
        '-t': 'n', '-a': '4', '-T': './tmp/'}

N = len(argv)
for i in range(1, N):
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
    qry, coverage, identity, norm, tmpdir, cpu, tmpsrt = args[
        '-i'], float(args['-c']), float(args['-y']), args['-n'], args['-t'], int(args['-a']), args['-T']
except:
    manual_print()
    raise SystemExit()


# make tmp dir for sort command
if tmpsrt != '/tmp/' or tmpsrt != '/tmp':
    os.system('mkdir -p %s' % tmpsrt)


#qry = sys.argv[1]
qry = os.path.abspath(qry)
fn = qry.split(os.sep)[-1]
os.system('mkdir -p %s_tmp/' % qry)
os.system('ln -sf %s %s_tmp/' % (qry, qry))
qry = qry + '_tmp/' + fn

# blast parser, return list contains blast results with the same query id
# remove the duplicated pairs or qid-sid


def blastparse0(f, coverage=.5, identity=0., norm='no', len_dict={}):
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
        idy, aln, mis, gop, qst, qed, sst, sed, evalue, score = list(
            map(float, j[2:12]))
        # the fastclust seq search format
        if len(j) > 13:
            qln, sln = list(map(float, j[12:14]))
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

        qcv = (1. + abs(qed - qst)) / qln
        scv = (1. + abs(sed - sst)) / sln
        if qcv < coverage or scv < coverage or idy < identity:
            continue

        if flag != qid:
            if output:
                yield list(output.values())

            mbsc = score
            # print 'max bit score is', mbsc, qid, sid
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

            if key not in output or output[key][-1] < Score:
                output[key] = [qid, sid, Score]

    if output:
        yield list(output.values())


# parse blast -m8 format (12 cols) or swiftOrtho -sc format (16 cols)
def blastparse(f, coverage=.5, identity=0., norm='no'):
    output = {}
    len_dict = {}
    flag = None
    # max bit score
    #mbsc = -1
    mbs_dict = {}
    for i in f:
        j = i[: -1].split('\t')
        # if len(j) != 12 or len(j) != 16:
        #    continue

        qid, sid = j[:2]
        qtx, stx = qid.split('|')[0], sid.split('|')[0]
        key = sid
        try:
            idy, aln, mis, gop, qst, qed, sst, sed, evalue, score = list(
                map(float, j[2:12]))
        except:
            continue
        # the fastclust seq search format
        if len(j) > 13:
            try:
                qln, sln = list(map(float, j[12:14]))
            except:
                continue
        else:
            if qid in len_dict:
                qln = len_dict[qid]
            else:
                qln = max(qst, qed)
                len_dict[qid] = qln

        qcv = (1. + abs(qed - qst)) / qln
        # if qcv<coverage or scv<coverage or idy<identity:
        if qcv < coverage or idy < identity:
            continue

        if flag != qid:
            if output:
                yield list(output.values())

            # print 'max bit score is', mbsc, qid, sid
            output = {}
            length = aln
            flag = qid
            if norm == 'bsr':
                if qid not in mbsc_dict:
                    mbsc_dict[qid] = score
                mbsc = mbsc_dict[qid]
                Score = score / mbsc
            elif norm == 'bal':
                Score = score / aln
            else:
                Score = score
            output[key] = [qid, sid, Score]

        else:
            if norm == 'bsr':

                if qid not in mbsc_dict:
                    mbsc_dict[qid] = score
                mbsc = mbsc_dict[qid]
                Score = score / mbsc
            elif norm == 'bal':
                Score = score / aln
            else:
                Score = score

            if key not in output or output[key][-1] < Score:
                output[key] = [qid, sid, Score]

    if output:
        yield list(output.values())


# distinguish IP and O
# return the IP and O
def get_IPO0(hits, l2n={}):
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
    ips, ots, cos = [], [], []
    for hit in hits:
        qid, sid, sco = hit
        if sid not in l2n:
            continue
        x, y = list(map(l2n.get, [qid, sid]))
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
                # ips.append(hit)
                ips.append(out)
            else:
                continue
        else:
            if sco >= sco_max[stx]:
                # ots.append(hit)
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

# get qIP, qOT and qCO


def get_qIPO(hits):
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
    ips, ots, cos = [], [], []
    for hit in hits:
        qid, sid, sco = hit
        sco = float(sco)
        if sid in visit:
            continue
        else:
            visit.add(sid)
        qtx = qid.split('|')[0]
        stx = sid.split('|')[0]
        qid, sid = qid < sid and [qid, sid] or [sid, qid]
        out = [qid, sid, sco]
        out = '\t'.join([qid, sid, str(sco)]) + '\n'
        if qtx == stx:
            if sco >= out_max and qid != sid:
                # if sco >= out_max:

                ips.append(out)
                outr = '\t'.join([sid, qid, str(sco)]) + '\n'
                ips.append(outr)
            else:
                continue
        else:
            if sco >= sco_max[stx]:
                ots.append(out)
            else:
                cos.append(out)

    # return IPs, OTs, COs
    return ips, ots, cos

# get IP and OT


def get_IPO(f):
    flag = None
    output = []
    for i in f:
        j = i[:-1].split('\t')
        qid, sid, score = j
        if flag != j[:2]:
            if len(output) == 4:
                yield output[0], output[1], sum(output[2:4]) / 2., 1
            elif len(output) == 3:
                yield output[0], output[1], output[2], 0
            else:
                # continue
                pass
            flag = j[:2]
            output = [qid, sid, float(score)]
        else:
            output.append(float(score))

    if len(output) == 4:
        # yield output[0], output[1], sum(output[2:4]) / 2., 1
        yield output[0], output[1], max(output[2:4]), 1
    elif len(output) == 3:
        yield output[0], output[1], output[2], 0
    else:
        pass


# parse and find IP, O from blast results
f = open(qry, 'r')
qip = qry + '.qIPs.txt'
_oqips = open(qip, 'w')

qot = qry + '.qOTs.txt'
_oqots = open(qot, 'w')

qco = qry + '.qCOs.txt'
_oqcos = open(qco, 'w')

# for i in blastparse(f, coverage, identity, norm, len_dict):
for i in blastparse(f, coverage, identity, norm):
    IPs, OTs, COs = get_qIPO(i)
    # print IPs, OTs, COs, l2n
    _oqips.writelines(IPs)
    _oqots.writelines(OTs)
    _oqcos.writelines(COs)

_oqips.close()
_oqots.close()
_oqcos.close()

# correct search results


def correct(s, m, l=None, r=None, sep=b'\n'):
    # sep=sep.encode()
    if not l and not r:
        return s.rfind(sep, 0, m) + 1
    M = s.rfind(sep, l, m) + 1
    if l < M < r:
        return M
    else:
        M = s.find(sep, m, r) + 1
        return M


def binary_search(s, p, key=lambda x: x.split('\t', 1)[0], L=0, R=-1, sep='\n'):
    #mx = chr(255)
    sep = sep.encode()
    if type(p) == str:
        p = p.encode()
    n = len(s)
    #pn = len(p)
    R = R == -1 and n - 1 or R
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
        #print(pat, p, type(p)==str, type(p.encode()))
        #print(pat, p)
        if pat >= p:
            r = m
        else:
            l = m

    left = m - 1
    while left >= 0:
        start = s.rfind(sep, 0, left)
        line = s[start + 1: left]
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
        # if key(s[right: end]) == p:
        if target == p:
            right = end + 1
        else:
            break

    pairs = s[left: right].strip().split(sep)
    return left, right, pairs


###############################################################################
# get OTs
###############################################################################
#inots = [-1] * len(l2n)
inots = set()
# sort qots
qotsrt = qot + '.srt'
os.system('export LC_ALL=C && sort -T %s --parallel=%s %s -o %s && rm %s' %
          (tmpsrt, cpu, qot, qotsrt, qot))

ots = qry + '.OTs.txt'
_oots = open(ots, 'w')
f = open(qotsrt, 'r')
for qid, sid, sco, lab in get_IPO(f):
    if lab == 1:
        out = '\t'.join([qid, sid, str(sco)]) + '\n'
        _oots.write(out)
        inots.add(qid)
        inots.add(sid)
    else:
        continue

_oots.close()
f.close()
os.system('rm %s' % qotsrt)


###############################################################################
# get IPs
###############################################################################
qipsrt = qip + '.srt'
os.system('export LC_ALL=C && sort -T %s --parallel=%s %s -o %s && rm %s' %
          (tmpsrt, cpu, qip, qipsrt, qip))

ipqa = {}
IPqA = {}
ips = qry + '.IPs.txt'
_oips = open(ips, 'w')

f = open(qipsrt, 'r')
for qid, sid, sco, lab in get_IPO(f):
    if lab == 1:
        out = '\t'.join([qid, sid, str(sco)]) + '\n'
        _oips.write(out)
        qtx = qid.split('|')[0]
        if qid < sid:
            if qid in inots or sid in inots:
                try:
                    ipqa[qtx][0] += float(sco)
                    ipqa[qtx][1] += 1.
                except:
                    ipqa[qtx] = [float(sco), 1.]

            try:
                IPqA[qtx][0] += float(sco)
                IPqA[qtx][1] += 1.
            except:
                IPqA[qtx] = [float(sco), 1.]

    else:
        continue

_oips.close()
f.close()
os.system('rm %s' % qipsrt)

for k in IPqA:
    a, b = k in ipqa and ipqa[k] or IPqA[k]
    IPqA[k] = a / b

#raise SystemExit()

###############################################################################
# get COs
###############################################################################
qcosrt = qco + '.srt'
os.system('export LC_ALL=C && sort -T %s --parallel=%s %s -o %s && rm %s' %
          (tmpsrt, cpu, qco, qcosrt, qco))


cos = qry + '.COs.txt'
_ocos = open(cos, 'w')

fqcosrt = open(qcosrt, 'rb')
try:
    Sqco = mmap(fqcosrt.fileno(), 0, access=ACCESS_READ)
except:
    Sqco = ''

fips = open(ips, 'rb')
try:
    Sips = mmap(fips.fileno(), 0, access=ACCESS_READ)
except:
    Sips = ''


f = open(ots, 'r')
for i in f:
    if not Sips or not Sqco:
        break
    # get ot pair
    #print(i, str(i).split('\t'))
    qid, sid, sco = i.split('\t')[:3]
    #qid, sid = map(int, [qid, sid])
    # get ip of ot
    # print(type(b'\t'))
    st, ed, qpairs = binary_search(Sips, qid, lambda x: x.split(b'\t', 2)[0])
    qips = [elem.split(b'\t')[1] for elem in qpairs]
    st, ed, spairs = binary_search(Sips, sid, lambda x: x.split(b'\t', 2)[0])
    sips = [elem.split(b'\t')[1] for elem in spairs]
    if not qpairs and not spairs:
        continue
    qips.append(qid.encode())
    sips.append(sid.encode())
    visit = set()
    for qip in qips:
        for sip in sips:
            if qip != qid or sip != sid:
                if (qip, sip) not in visit:
                    visit.add((qip, sip))
                else:
                    continue

                st, ed, pairs = binary_search(
                    Sqco, [qip, sip], lambda x: x.split(b'\t', 3)[:2])
                if pairs:
                    xyzs = [elem.split(b'\t') for elem in pairs]
                    x, y = xyzs[0][:2]
                    sco = max([float(elem[2]) for elem in xyzs])
                    #print(x.decode(), y.decode(), sco)
                    _ocos.write(
                        '\t'.join([x.decode(), y.decode(), str(sco)]) + '\n')
                    #_ocos.write(pairs[0]+'\n')
                else:
                    continue

_ocos.close()
f.close()
os.system('rm %s' % qcosrt)


###############################################################################
# print normalized IPs
###############################################################################
f = open(ips, 'r')
for i in f:
    # print 'all_IP\t' + i[:-1]
    #x, y, score = i[:-1].split('\t')
    #x, y = map(int, [x, y])
    #qid, sid = n2l[x], n2l[y]
    qid, sid, score = i[:-1].split('\t')
    if qid >= sid:
        continue
    tax = qid.split('|')[0]
    avg = IPqA[tax]
    score = float(score)
    try:
        out = list(map(str, ['IP', qid, sid, score / avg]))
    except:
        continue
    print('\t'.join(out))

f.close()
IPqA.clear()


# get co or ot from same taxon
def get_sam_tax0(f, n2l):
    flag = None
    out = []
    visit = set()
    for i in f:
        x, y, sco = i[:-1].split('\t')
        x, y = list(map(int, [x, y]))

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

# get orthology relationship with same tax name


def get_sam_tax(f):
    flag = None
    out = []
    visit = set()
    for i in f:
        qid, sid, sco = i[:-1].split('\t')
        qtx = qid.split('|')[0]
        sco = float(sco)
        if qtx != flag:
            if out:
                yield out
            flag = qtx
            out = [[qid, sid, sco]]
            visit = set((qid, sid))
        else:
            if (qid, sid) not in visit:
                out.append([qid, sid, sco])
                visit.add((qid, sid))
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
for i in get_sam_tax(f):
    for j in n_co_ot(i):
        out = '\t'.join(map(str, j))
        print('OT\t' + out)

f.close()

f = open(cos, 'r')
for i in get_sam_tax(f):
    for j in n_co_ot(i):
        out = '\t'.join(map(str, j))
        print('CO\t' + out)

f.close()


if tmpdir == 'n':
    os.system('rm -rf %s_tmp/' % qry)

if tmpsrt != '/tmp/' or tmpsrt != '/tmp':
    os.system('rm -rf %s' % tmpsrt)
