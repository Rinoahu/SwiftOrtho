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
    qry, coverage, identity, norm, tmpdir, np = args['-i'], float(args['-c']), float(args['-y']), args['-n'], args['-t'], int(args['-a'])

except:
    manual_print()
    raise SystemExit()


# mean of list/array
mean = lambda x: float(sum(x)) / len(x)


# the ortholog class
class OTH:
    def __init__(self, fn=None, cov=.95, idy=0, evl=1e-5, norm='bal'):
        self.fn = fn
        self.cov = cov
        self.idy = idy
        self.evl = evl
        self.norm = norm


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
        f = open(self.fn, 'r')
        for hits in self.parse(f):
            qid, sid, idy, aln, mis, gop, qst, qed, sst, sed, evl, sco = hits[0]
            mxb = sco
            qln = max(qst, qed)
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
        flag = 0
        ips, ots, cos = [], {}, []
        for hit in hits:
            qid, sid, sco = hit
            qtx = qid.split('|')[0]
            stx = sid.split('|')[0]
            if qtx == stx:
                if flag == 0:
                    ips.append(hit)
                #else:
                #    cos.append(hit)
            else:
                flag = 1
                if stx not in ots:
                    ots[stx] = hit
                else:
                    cos.append(hit)

        ots = ots.values()
        return ips, ots, cos


    # get the range of a given qid
    def get_range(self, loc, qids):
        i = self.l2i.get(loc)
        if i:
            start, end = qids[i:i+2]
        else:
            start = end = -1
        return start, end

    # given qid and sid, return index
    def get_pair(self, qid, qids, sid, sids):
        #x, y = map(self.loc2idx, [qid, sid])
        if qid in self.l2i and sid in self.l2i:
            x, y  = map(self.l2i.get, [qid, sid])
        else:
            return -1
        x, y = x <= y and [x, y] or [y, x]
        #print 'x', x, y, len(qids), qids[x: x+2], map(len, [self.qco_qids, self.qot_qids, self.qip_qids])
        start, end = qids[x:x+2]
        if start == end or not sids:
            return -1
        else:
            Y = bisect_left(sids, y, start, end-1)
            return y == sids[Y] and Y or -1



    # add hits into qids, sids and scos
    def add_hits(self, qids, sids, scos, hits):
        #[[self.loc2idx(elem[0]), self.loc2idx(elem[1]), elem[2]] for elem in hits]
        new = []
        for qid,sid,sco in hits:
            if qid not in self.l2i or sid not in self.l2i:
                continue
            x, y = map(self.l2i.get, [qid, sid])
            if x < y:
                new.append([y, -sco])

            elif x > y:
                z = self.get_pair(qid, qids, sid, sids)
                if z != -1:
                    #print 'oh yeah'
                    scos[z] = (sco - scos[z])/2.
            else:
                continue

        
        #print 'new', len(new), new, len(sids)
        new.sort()
        sids.extend([elem[0] for elem in new])
        scos.extend([elem[1] for elem in new])
        qids.append(len(sids))


    # identify (co-)orthologs and paralogs
    def fit(self, fn=None):
        if fn != None:
            self.fn = fn
        # filter by coverage and idy
        flag = None
        self.qip_qids = [0]
        self.qip_sids = []
        self.qip_scos = []
        self.qot_qids = [0]
        self.qot_sids = []
        self.qot_scos = []
        self.qco_qids = [0]
        self.qco_sids = []
        self.qco_scos = []
        qips = []
        qots = []
        qcos = []
        self.loc = []
        #idx = 0
        f = open(self.fn, 'r')
        for hits in self.filter():
            qid = hits[0][0]
            self.loc.append(qid)

        self.N = len(self.loc)
        self.l2i = {}
        for i in xrange(self.N):
            self.l2i[self.loc[i]] = i


        for hits in self.filter():
            qid = hits[0][0]
            qip_hits, qot_hits, cos_hits = self.get_qico(hits)
            self.add_hits(self.qip_qids, self.qip_sids, self.qip_scos, qip_hits)
            self.add_hits(self.qot_qids, self.qot_sids, self.qot_scos, qot_hits)
            self.add_hits(self.qco_qids, self.qco_sids, self.qco_scos, cos_hits)
 

        f.close()

 

    # get orthologs
    def get_ots(self):
        for qid in self.loc:
            start, end = self.get_range(qid, self.qot_qids)
            for i in xrange(start, end):
                sco = self.qot_scos[i]
                if sco >= 0:
                    sid = self.loc[self.qot_sids[i]]
                    yield qid, sid, sco

    # get paralogs
    def get_ips(self):
        for qid in self.loc:
            start, end = self.get_range(qid, self.qip_qids)
            for i in xrange(start, end):
                sco = self.qip_scos[i]
                if sco >= 0:
                    sid = self.loc[self.qip_sids[i]]
                    yield qid, sid, sco
               

    # get co by given a ortholog
    def get_cos(self):
        for qid, sid, sco in self.get_ots():
            qips = [qid]
            start, end = self.get_range(qid, self.qip_qids)
            qips.extend([self.loc[self.qip_sids[elem]] for elem in xrange(start, end)])

            sips = [sid]
            start, end = self.get_range(sid, self.qip_qids)
            sips.extend([self.loc[self.qip_sids[elem]] for elem in xrange(start, end)])

            #print qid, qips, sid, sips
            qco_visit = [0] * len(self.qco_scos)
            qot_visit = [0] * len(self.qot_scos)

            for qip in qips:
                for sip in sips: 
                    #if qip == qid and sip == sid:
                    #    continue
                    i = self.get_pair(qip, self.qco_qids, sip, self.qco_sids)
                    if i != -1 and qco_visit[i] == 0:
                        yield qid, sid, abs(self.qco_scos[i])
                        qco_visit[i] = 1
                    else:
                        i = self.get_pair(qip, self.qot_qids, sip, self.qot_sids)
                        if i != -1 and qot_visit[i] == 0:
                            if self.qot_scos[i] < 0:
                                yield qid, sid, abs(self.qot_scos[i])
                            qot_visit[i] = 1
                        else:
                            continue

    # normalization of inparalog, ortholog and co-ortholog
    def get_norm_ips(self):
        N = len(self.loc)
        ots = [0] * N
        for i in xrange(N):
            qid = self.loc[i]
            start, end = self.get_range(qid, self.qot_qids)
            for j in xrange(start, end):
                if self.qot_scos[j] >= 0:
                    ots[i] = 1
                    break

        flag = None
        ips_t, ips_a = [], []
        for qid, sid, sco in self.get_ips():
            qtx = qid.split('|')[0]
            x, y = map(self.l2i.get, [qid, sid])
            if qtx != flag:
                if ips_t:
                    avg = ips_a and mean([elem[2] for elem in ips_a]) or mean([elem[2] for elem in ips_t])
                    for a, b, c in ips_t:
                        yield a, b, c/avg

                # pass
                flag = qtx
                ips_t= [[qid, sid, sco]]
                #x, y = map(self.l2i.get, [qid, sid])
                if ots[x] == 1 or ots[y] == 1:
                    ips_a = [[qid, sid, sco]]
                else:
                    ips_a = []

            else:
                ips_t.append([qid, sid, sco])
                #x, y = map(self.l2i, [qid, sid])
                if ots[x] == 1 or ots[y] == 1:
                    ips_a.append([qid, sid, sco])
                    

        if ips_t:
            #avg = mean([elem[2] for elem in ips_a])
            #ipqa_avg = avg > 0 and avg or mean([elem[2] for elem in ips_t])
            avg = ips_a and mean([elem[2] for elem in ips_a]) or mean([elem[2] for elem in ips_t])
            for a, b, c in ips_t:
                yield a, b, c/avg


    def get_norm_ots(self):
        flag = None
        out = {}
        for qid, sid, sco in self.get_ots():
            qtx = qid.split('|')[0]
            stx = sid.split('|')[0]
            if flag != qtx:
                if out:
                    for i in out.values():
                        #avg = sum([elem[2] for elem in i])*1./len(i)
                        avg = mean([elem[2] for elem in i])
                        for a, b, c in i:
                            yield a, b, c/avg

                flag = qtx
                out = {stx: [[qid, sid, sco]]}
            else:
                try:
                    out[stx].append([qid, sid, sco])
                except:
                    out[stx] = [[qid, sid, sco]]

        if out:
            for i in out.values():
                #avg = sum([elem[2] for elem in i])*1./len(i)
                avg = mean([elem[2] for elem in i])
                for a, b, c in i:
                    yield a, b, c/avg


    def get_norm_cos(self):
        flag = None
        out = {}
        for qid, sid, sco in self.get_cos():
            qtx = qid.split('|')[0]
            stx = sid.split('|')[0]
            if flag != qtx:
                if out:
                    for i in out.values():
                        #avg = sum([elem[2] for elem in i])*1./len(i)
                        avg = mean([elem[2] for elem in i])
                        for a, b, c in i:
                            yield a, b, c/avg

                flag = qtx
                out = {stx: [[qid, sid, sco]]}
            else:
                try:
                    out[stx].append([qid, sid, sco])
                except:
                    out[stx] = [[qid, sid, sco]]

        if out:
            for i in out.values():
                #avg = sum([elem[2] for elem in i])*1./len(i)
                avg = mean([elem[2] for elem in i])
                for a, b, c in i:
                    yield a, b, c/avg




clf = OTH(qry)
clf.fit()
#for i in clf.get_ips():
#    print i

for i in clf.get_norm_ips():
    print i



#for i in clf.get_norm_ots():
#    print i

#for i in clf.get_ots():
#    print i



#for i in clf.get_cos():
#    print i

#for i in clf.get_norm_cos():
#    print i


