#!usr/bin/env python
import sys
import networkx as nx
from collections import Counter

# input is m8 file
qry = sys.argv[1]

id2n = {}
n2id = {}
G = nx.Graph()

f = open(qry, 'r')
flag = 0
for i in f:
    j = i.split('\t')
    qid, sid = j[:2]
    if qid not in id2n:
        id2n[qid] = flag
        n2id[flag] = qid
        flag += 1

    if sid not in id2n:
        id2n[sid] = flag
        n2id[flag] = sid
        flag += 1
       
    qn, sn = map(id2n.get, [qid, sid])
    if not G.has_edge(qn, sn):
       G.add_edge(qn, sn)

f.close()

def get_com(G):
    n2c = []
    for g in nx.connected_component_subgraphs(G):
        n2c.extend(g.nodes())
        flag += 1


f = open(qry, 'r')
for i in f:

