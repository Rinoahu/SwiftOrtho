#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#  Copyright © xh
# CreateTime: 2016-06-03 13:55:13

# this is new fast and memory efficient version of orthomcl algorithm
import sys
import networkx as nx
from math import log10
import os
from commands import getoutput
from mmap import mmap, ACCESS_WRITE, ACCESS_READ
from collections import Counter
from bisect import bisect_left
import json
from leveldict import LevelDict, LevelDictSerialized
import shelve


# initial a dictionary
#Dict = lambda x: LevelDictSerialized(x, serializer=json, max_open_files=25, block_cache_size=1024**2*512, write_buffer_size=1024**2*512)
Dict = lambda x: LevelDictSerialized(x, serializer=json, max_open_files=50)

class Dict0:
    def __init__(self, name, n = 43):
        self.N = n
        os.system('mkdir -p %s_db'%name)
        self.temp = name + '_db'
        #self.dbs = [anydbm.open(self.temp + '/%d'%elem, 'c') for elem in xrange(self.N)]
        self.dbs = [shelve.open(self.temp + '/%d'%elem) for elem in xrange(self.N)]

    def __setitem__(self, x, y):
        i = hash(x) % self.N
        self.dbs[i][x] = y

    def __getitem__(self, x):
        i = hash(x) % self.N
        return self.dbs[i][x]

    def get(self, x, y = None):
        i = hash(x) % self.N
        return self.dbs[i].get(x, y)

    def __iter__(self):
        for db in self.dbs:
            for i in db:
                yield i

    def __len__(self):
        return sum(map(len, self.dbs))

    def update(self, x):
        for k, v in x.iteritems():
            self[k] = v

    def close(self):
        for i in self.dbs:
            i.close()

    def clear(self):
        self.close()
        os.system('rm -rf %s'%self.temp)


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
    n = len(s)
    pn = len(p)
    R = R == -1 and n-1 or R
    l = correct(s, L, sep=sep)
    r = correct(s, R, sep=sep)
    # find left
    while l < r:
        m = (l + r) // 2
        m = correct(s, m, l, r, sep=sep)
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
        #if key(line).startswith(p):
        if key(line) == p:
            left = start
        else:
            break
    left += 1
    line = s[left: s.find('\n', left)]
    #if not key(line).startswith(p):
    if key(line) != p:
        return -1, -1, []

    right = left
    while 1:
        end = s.find('\n', right)
        #if key(s[right: end]).startswith(p):
        if key(s[right: end]) == p:
            right = end + 1
        else:
            break

    pairs = s[left: right].strip().split('\n')
    return left, right, pairs



# print the manual
def manual_print():
    print 'Usage:'
    print '    python find_orth_ultra.py -i foo.sc [-c .5] [-y 50] [-n no]'
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


# mean of list/array
mean = lambda x: float(sum(x)) / len(x)


# the ortholog class
class OTH:
    def __init__(self, fn=None, cov=.5, idy=0, evl=1e-5, norm='bal'):
        self.fn = fn
        self.cov = cov
        self.idy = idy
        self.evl = evl
        self.norm = norm
        self.temp = fn + '_tmp/'
        os.system('rm -rf %s'%self.temp)
        os.system('mkdir -p %s'%self.temp)

    # parse blast8
    def parse(self, f):
        flag = None
        hits = {}
        for i in f:
            j = i[:-1].split('\t')
            qid, sid = j[:2]
            idy, aln, mis, gop, qst, qed, sst, sed, evl, sco = map(float, j[2:12])
            if qid != flag:
                if hits:
                    yield hits.values()
                flag = qid
                hits = {sid: [qid, sid, idy, aln, mis, gop, qst, qed, sst, sed, evl, sco]}
            else:
                if sid not in hits:
                    hits[sid] = [qid, sid, idy, aln, mis, gop, qst, qed, sst, sed, evl, sco] 

        if hits:
            yield hits.values()


    # filter the blast8 hits by identity, query-coverage and e-value
    def filter(self):
        qln_dict = {}
        f = open(self.fn, 'r')
        for hits in self.parse(f):
            qid, sid, idy, aln, mis, gop, qst, qed, sst, sed, evl, sco = hits[0]
            mxb = sco
            #qln = max(qst, qed)
            x = max(qst, qed)
            qln_dict[qid] = max(qln_dict.get(qid, 0), x)
            qln = qln_dict[qid]
            out = []
            for hit in hits:
                qid, sid, idy, aln, mis, gop, qst, qed, sst, sed, evl, sco = hit
                qcv = (abs(qed-qst)+1.)/qln
                if idy < self.idy or qcv < self.cov:
                    continue

                if self.norm == 'bsr':
                    sco /= mxb
                elif self.norm == 'bsl':
                    sco /= aln
                else:
                    pass
                out.append([qid, sid, sco])
            if out:
                yield out

    # get Q_inparalog and Q_(co-)ortholog
    def get_qico(self, hits):
        # get max of each species
        sco_max = Counter()
        out_max = 0
        for qid, sid, sco in hits:
            qtx = qid.split('|')[0]
            stx = sid.split('|')[0]
            sco_max[stx] = max(sco_max[stx], sco)
            if qtx != stx:
                out_max = max(out_max, sco)

        visit = set()
        ips, ots, cos = [], [], []
        for hit in hits:
            qid, sid, sco = hit
            if sid in visit:
                continue
            else:
                visit.add(sid)
            qtx = qid.split('|')[0]
            stx = sid.split('|')[0]
            #if qtx == stx and sco >= out_max:
            if qtx == stx:
                if sco >= out_max:
                    ips.append(hit)
                else:
                    continue
                    #cos.append(hit)
            else:
                if sco >= sco_max[stx]:
                    ots.append(hit)
                else:
                    cos.append(hit)

            #elif qtx != stx and sco >= sco_max[stx]:
            #    ots.append(hit)
            #else:
            #    cos.append(hit)

        return ips, ots, cos

    # get score from string
    def get(self, s, p):
        st, ed, pairs = binary_search(s, p)
        return [elem.split('\t') for elem in pairs] 

    # get list from string
    def getlist(self, s):
        return [elem.split('\t') for elem in s.split('\t') if elem]

    # identify (co-)orthologs and paralogs
    def fit(self, fn=None):
        if fn != None:
            self.fn = fn
        # filter by coverage and idy
        IPs = Dict(self.temp + './ips')
        IPss = Dict(self.temp + './ipss')
        OTs = Dict(self.temp + './ots')
        OTs_avg = Dict(self.temp + './ots_avg')
        COs = Dict(self.temp + './cos')
        COs_avg = Dict(self.temp + './cos_avg')
        INOTs = set()
        # get qIP, qOT and qCO
        for hits in self.filter():
            qip_hits, qot_hits, qco_hits = self.get_qico(hits)
            out = []
            for qid, sid, sco in qip_hits:
                key = qid < sid and qid + '\t' + sid or sid + '\t' + qid
                if key in IPs:
                    old = float(IPs[key])
                    if old < 0:
                        sc = (sco - old) / 2.
                        IPs[key] = sc
                else:
                    IPs[key] = str(-sco)
                if out:
                    out.append(sid)
                else:
                    out = [qid, sid]
            if out:
                IPss[out[0]] = '\t'.join(out)

            for qid, sid, sco in qot_hits:
                key = qid < sid and qid + '\t' + sid or sid + '\t' + qid
                if key in OTs:
                    old = float(OTs[key])
                    if old < 0:
                        OTs[key] = (sco - old) / 2.
                        qtx = qid.split('|')[0]
                        stx = sid.split('|')[0]
                        k1 = qtx < stx and qtx + '\t' + stx or stx + '\t' + qtx
                        if k1 not in OTs_avg:
                            OTs_avg[k1] = '\t'.join(map(str, [sco, 1.]))
                        else:
                            a, b = map(float, OTs_avg[k1].split('\t'))
                            OTs_avg[k1] = '\t'.join(map(str, [a+sco, b+1.]))

                        INOTs.add(qid)
                        INOTs.add(sid)

                else:
                    OTs[key] = str(-sco)

            for qid, sid, sco in qco_hits:
                key = qid < sid and qid + '\t' + sid or sid + '\t' + qid
                if key in COs:
                    old = float(COs[key])
                    if old < 0:
                        COs[key] = (old - sco) / 2.
                else:
                    COs[key] = str(-sco)

            for key in OTs:
                qid, sid = key.split('\t')[:2]
                QIP, SIP = IPss.get(qid, '').split('\t'), IPss.get(sid, '').split('\t')
                for qip in QIP:
                    for sip in SIP:
                        if qip and sip:
                            key = qip < sip and qip + '\t' + sip or sip + '\t' + qip
                            if key in COs:
                                sco = COs[key]
                                COs[key] = abs(float(sco))
                                qtx = qip.split('|')[0]
                                stx = sip.split('|')[0]
                                key = qtx < stx and qtx + '\t' + stx or stx + '\t' + qtx
                                if k1 not in OTs_avg:
                                    COs_avg[k1] = '\t'.join(map(str, [sco, 1.]))
                                else:
                                    a, b = map(float, OTs_avg[k1].split('\t'))
                                    COs_avg[k1] = '\t'.join(map(str, [a+float(sco), b+1.]))

        for k in OTs_avg:
            a, b = map(float, OTs_avg[k].split('\t'))
            OTs_avg[k] = a / b

        for k in COs_avg:
            a, b = map(float, COs_avg[k].split('\t'))
            COs_avg[k] = a / b


        IPqA = {}
        IPA = {}
        for k in IPs:
            qid, sid = k.split('\t')[:2]
            v = float(IPs[k])
            qtx = k.split('|')[0]
            if v > 0:
                try:
                    IPqA[qtx][0] += v
                    IPqA[qtx][1] += 1.
                except:
                    IPqA[qtx] = [v, 1.]
            if qid in INOTs or sid in INOTs: 
                try:
                    IPA[qtx][0] += v
                    IPA[qtx][1] += 1.
                except:
                    IPA[qtx] = [v, 1.]
        for k in IPqA:
            a, b = IPqA[k]
            if k in IPA:
                a, b = IPA[k]
            IPqA[k] = a / b

        self.IPqA = IPqA
        self.COs_avg = COs_avg
        self.OTs_avg = OTs_avg
        self.IPs = IPs
        self.OTs = OTs
        self.COs = COs

    # print all ortholog and paralog
    def printf(self):
        for i in self.IPs:
            sco = float(self.IPs[i])
            if sco > 0:
                qtx = i.split('|', 2)[0]
                nsc = sco / self.IPqA[qtx]
                print i + '\t' + str(nsc)

        for i in self.OTs:
            sco = float(self.OTs[i])
            if sco > 0:
                key = qtx < stx and qtx + '\t' + stx or stx + '\t' + qtx
                nsc = sco / self.OTs_avg[key]
                print i + '\t'  + str(nsc)

        for i in self.COs:
            sco = float(self.COs[i])
            if sco > 0:
                key = qtx < stx and qtx + '\t' + stx or stx + '\t' + qtx
                nsc = sco / self.COs_avg[key]
                print i + '\t'  + str(nsc)

clf = OTH(qry, coverage, identity)
clf.fit()
clf.printf()
