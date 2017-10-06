#!usr/bin/env python
# get rbh from blast -m8 results

import sys

if len(sys.argv[1:]) < 1:
    print 'python this.py foo.blast8'
    raise SystemExit()

def blastparse(f):
    flag = None
    out = []
    for i in f:
        j = i.split('\t')
        qid, sid = j[:2]
        sco = int(j[11])
        if qid != flag:
            if out:
                yield out
            out = [[qid, sid, sco]]
            flag = qid
        else:
            out.append([qid, sid, sco])

    if out:
        yield out

# find rbh
def get_rbh(hits):
    out = {}
    for hit in hits:
        qid, sid, sco = hit
        qtx = qid.split('|')[0]
        stx = sid.split('|')[0]
        if qtx == stx:
            continue

        if stx in out:
            if out[stx][2] < sco:
                out[stx] = [qid, sid, sco]
        else:
            out[stx] = [qid, sid, sco]

    return out.values()



rbh_dict = set()

qry = sys.argv[1]
f = open(qry, 'r')

for hits in blastparse(f):
    for qid, sid, sco in get_rbh(hits):
        if qid > sid:
            qid, sid = sid, qid
        key = qid + '\t' + sid
        if key in rbh_dict:
            print key
            rbh_dict.remove(key)
        else:
            rbh_dict.add(key)

f.close()



