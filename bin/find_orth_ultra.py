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

# initial a dictionary
Dict = lambda x: LevelDictSerialized(x, serializer=json)


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


    # identify (co-)orthologs and paralogs
    def fit(self, fn=None):
        if fn != None:
            self.fn = fn
        # filter by coverage and idy
        qIPs = Dict(self.temp + './qips')
        qOTs = Dict(self.temp + './qots')
        qCOs = Dict(self.temp + './qcos')

        # get qIP, qOT and qCO
        for hits in self.filter():
            qip_hits, qot_hits, qco_hits = self.get_qico(hits)
            out = {}
            for x, y, z in qip_hits:
                try:
                    out[x][y] = z
                except:
                    out[x] = {y: z}
            qIPs.update(out)

            out = {}
            for x, y, z in qot_hits:
                try:
                    out[x][y] = z
                except:
                    out[x] = {y: z}
            qOTs.update(out)

            out = {}
            for x, y, z in qco_hits:
                try:
                    out[x][y] = z
                except:
                    out[x] = {y: z}
            qCOs.update(out)


        # get IP, OT and CO
        OTs = Dict(self.temp + './ots')
        OTs_avg = Dict(self.temp + './ots_avg')

        # get orthologs
        #print 'qot', [x for x in qOTs.iteritems()]
        #for qid, sids in qOTs.iteritems():
        for qid in qOTs:
            sids = qOTs[qid]
            out = {}
            for sid, qsc in sids.iteritems():
                #if qid < sid:
                if 1:
                    ssc = qOTs.get(sid, {}).get(qid)
                    if ssc:
                        sco = (qsc+ssc) / 2.
                        try:
                            out[qid][sid] = sco
                        except:
                            out[qid] = {sid: sco}
            OTs.update(out)
            for qid in out:
                for sid, sco in out[qid].iteritems():
                    qtx = qid.split('|')[0]
                    stx = sid.split('|')[0]
                    key = qtx + '\t' + stx
                    if key not in OTs_avg:
                        OTs_avg[key] = [sco, 1.]
                    else:
                        a, b = OTs_avg[key]
                        OTs_avg[key] = [a+sco, b+1.]

        for key in OTs_avg:
            a, b = OTs_avg[key]
            OTs_avg[key] = a/b
        self.OTs_avg = OTs_avg

        # get inparalogs
        IPs = Dict(self.temp + './ips')
        IP_pairs = {}
        IPqA = {}
        #for qid, sids in qIPs.iteritems():
        for qid in qIPs:
            sids = qIPs[qid]
            out = {}
            for sid, qsc in sids.iteritems():
                #if qid < sid:
                if 1:
                    ssc = qIPs.get(sid, {}).get(qid)
                    if ssc:
                        sco = (qsc+ssc) / 2.
                        try:
                            out[qid][sid] = sco
                        except:
                            out[qid] = {sid: sco}

            IPs.update(out)
            for qid in out:
                sid, sco = out[qid].items()[0]
                qtx = qid.split('|')[0]
                stx = sid.split('|')[0]
                if qtx == stx:
                    if qtx in IP_pairs:
                        a, b = IP_pairs[qtx]
                        IP_pairs[qtx] = [a+sco, b+1.]
                    else:
                        IP_pairs[qtx] = [sco, 1.]
                if qid in OTs or sid in OTs:
                    if qtx in IPqA:
                        a, b = IPqA[qtx]
                        IPqA[qtx] = [a+sco, b+1.]
                    else:
                        IPqA[qtx] = [sco, 1.]

        self.IPqA_avg = {}
        for qtx in IP_pairs:
            if qtx in IPqA:
                x, y = IPqA[qtx]
            else:
                x, y = IP_pairs[qtx]
            self.IPqA_avg[qtx] = x / y 

        # get co-orthologs
        COs_avg = Dict(self.temp + './cos_avg')
        COs_visit = Dict(self.temp + './cos_visit')
        COs = Dict(self.temp + '.cos')
        for qid in OTs:
            for sid in OTs[qid].iterkeys():
                qips = qIPs.get(qid, {}).keys()
                sips = qIPs.get(sid, {}).keys()
                for qip in qips:
                    for sip in sips:
                        vk = qip + '\t' + sip
                        if vk in COs_visit:
                            continue
                        else:
                            COs_visit[vk] = 1

                        if qip != qid or sip != sid:
                            qsc = qCOs.get(qip, {}).get(sip)
                            ssc = qCOs.get(sip, {}).get(qip)
                            if qsc != None and ssc != None:
                                sco = (qsc + ssc) / 2.
                            elif qsc != None:
                                sco = qsc
                            elif ssc != None:
                                sco = ssc
                            else:
                                continue
                            k0 = qip + '\t' + sip
                            COs[k0] = sco
                            # avg of CO in across A and B
                            qtx = qip.split('|')[0]
                            stx = sip.split('|')[0]
                            key = qtx + '\t' + stx
                            if key not in COs_avg:
                                COs_avg[key] = [sco, 1.]
                                #print 'k0', k0, 'sco', sco, 'key', key, [sco, 1.]
                            else:
                                a, b = COs_avg[key]
                                COs_avg[key] = [a+sco, b+1.]
                                #try:
                                #    COs_avg[key] = [a+sco, b+1.]
                                #except:
                                #    print 'sco a b', COs_avg.items(), sco, a, b
                                #    raise SystemExit()


        for key in COs_avg:
            a, b = COs_avg[key]
            COs_avg[key] = a/b
        self.COs_avg = COs_avg

        self.IPs = IPs
        self.OTs = OTs
        self.COs = COs

        # print all ortholog and paralog
    def printf(self):
        for qid in self.IPs:
            for sid, sco in self.IPs[qid].iteritems():
                if qid > sid:
                    continue
                qtx = qid.split('|')[0]
                key = qtx
                nsc = sco / self.IPqA_avg[key]
                print '\t'.join(['IP'] + map(str, [qid, sid, nsc]))

        for qid in self.OTs:
            for sid, sco in self.OTs[qid].iteritems():
                if qid > sid:
                    continue
                qtx = qid.split('|')[0]
                stx = sid.split('|')[0]
                key = qtx + '\t' + stx
                nsc = sco / self.OTs_avg[key]
                print '\t'.join(['OT'] + map(str, [qid, sid, nsc]))

        #for key, sco in self.COs.iteritems():
        for key in self.COs:
            sco = self.COs[key]
            qid, sid = key.split('\t')
            qtx = qid.split('|')[0]
            stx = sid.split('|')[0]
            key = qtx + '\t' + stx
            nsc = sco / self.COs_avg[key]
            print '\t'.join(['OT'] + map(str, [qid, sid, nsc]))



clf = OTH(qry, coverage, identity)
clf.fit()
clf.printf()
