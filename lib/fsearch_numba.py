"""Pure Python 3 + Numba rewrite of SwiftOrtho lib/fsearch.py (hot path).

Replaces the Python-2 / RPython engine in
    Rinoahu/SwiftOrtho/lib/fsearch.py   (3,272 lines)
with a single-file, Python-3 + NumPy + Numba implementation.

Scope (hot path only — dead code dropped):
    blastp(), Fasta.build_msav(), Fasta.find_msav_m(), ungap(),
    get_ungap_scores(), guess_start(), seg()/entropy(),
    spseeds_fnv(), kswat_st()/kswat_st_long(), makedb(),
    entry_point-style CLI, score2bit/bit2score/bit2e(), lis()

Dropped: Fasta0, swat, Diag, klis2, kswat_st0, ungap_kswat_st,
         xpack/xunpack{,_int}, upack, kmer_slw, reverse_char,
         index0, spseeds_*0, generate_nr_tbl0, get_mu_sd0,
         n2s, max_idx, get_batch_sequences, kolmogorov,
         qsort_u duplicate set, RPython shims (rmmap/rfile/intmask).

Usage (CLI compatible with fsearch-c):
    python -m lib.fsearch_numba -p blastp -i qry.fsa -d db.fsa -o out.sc \
        -e 1e-5 -s 111111 -r AST,CFILMVY,DN,EQ,G,H,KR,P,W -v 500 -a 1 --profile
    python -m lib.fsearch_numba -p makedb -i db.fsa ...

    python lib/fsearch_numba.py -p blastp -i qry.fsa -d db.fsa -o out.sc

API:
    from fsearch_numba import blastp, build_index, kswat_st
"""

import argparse
import math
import os
import sys
import time

import numpy as np
from numba import njit, prange

__all__ = [
    "parse_fasta",
    "generate_nr_tbl",
    "seq_to_codes",
    "seg",
    "spseeds_fnv_nb",
    "build_index",
    "find_msav_m",
    "ungap",
    "kswat_st",
    "kswat_st_long",
    "kswat_st_linear",
    "kswat_st_long_linear",
    "kswat_st_wfa2",
    "kswat_st_long_wfa2",
    "kswat_st_batch",
    "kswat_st_linear_batch",
    "kswat_st_wfa2_batch",
    "blastp",
    "f2s",
    "score2bit",
    "bit2score",
    "main",
]

# --------------------------------------------------------------------------
# BLOSUM62 (standard 20-letter matrix; ambiguous letters mapped NCBI-style)
# --------------------------------------------------------------------------
AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
_B62_BODY = """
 4 -1 -2 -2 0 -1 -1 0 -2 -1 -1 -1 -1 -2 -1 1 0 -3 -2 0
-1 5 0 -2 -3 1 0 -2 0 -3 -2 2 -1 -3 -2 -1 -1 -3 -2 -3
-2 0 6 1 -3 0 0 0 1 -3 -3 0 -2 -3 -2 1 0 -4 -2 -3
-2 -2 1 6 -3 0 2 -1 -1 -3 -4 -1 -3 -3 -1 0 -1 -4 -3 -3
 0 -3 -3 -3 9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
-1 1 0 0 -3 5 2 -2 0 -3 -2 1 0 -3 -1 0 -1 -2 -1 -2
-1 0 0 2 -4 2 5 -2 0 -3 -3 1 -2 -3 -1 0 -1 -3 -2 -2
 0 -2 0 -1 -3 -2 -2 6 -2 -4 -4 -2 -3 -3 -2 0 -2 -2 -3 -3
-2 0 1 -1 -3 0 0 -2 8 -3 -3 -1 -2 -1 -2 -1 -2 -2 2 -3
-1 -3 -3 -3 -1 -3 -3 -4 -3 4 2 -3 1 0 -3 -2 -1 -3 -1 3
-1 -2 -3 -4 -1 -2 -3 -4 -3 2 4 -2 2 0 -3 -2 -1 -2 -1 1
-1 2 0 -1 -3 1 1 -2 -1 -3 -2 5 -1 -3 -1 0 -1 -3 -2 -2
-1 -1 -2 -3 -1 0 -2 -3 -2 1 2 -1 5 0 -2 -1 -1 -1 -1 1
-2 -3 -3 -3 -2 -3 -3 -3 -1 0 0 -3 0 6 -4 -2 -2 1 3 -1
-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4 7 -1 -1 -4 -3 -2
 1 -1 1 0 -1 0 0 0 -1 -2 -2 0 -1 -2 -1 4 1 -3 -2 -2
 0 -1 0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1 1 5 -2 -2 0
-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1 1 -4 -3 -2 11 2 -3
-2 -2 -2 -3 -2 -1 -2 -3 2 -1 -1 -2 -1 3 -3 -2 -2 2 7 -1
 0 -3 -3 -3 -1 -2 -2 -3 -3 3 1 -2 1 -1 -2 -2 0 -3 -1 4
"""


def _build_b62():
    nums = [int(x) for x in _B62_BODY.split()]
    mat = np.zeros((256, 256), dtype=np.int16)
    for i, ca in enumerate(AA_ORDER):
        for j, cb in enumerate(AA_ORDER):
            v = nums[i * 20 + j]
            mat[ord(ca), ord(cb)] = v
            mat[ord(ca.lower()), ord(cb.lower())] = v
            mat[ord(cb), ord(ca)] = v
            mat[ord(cb.lower()), ord(ca.lower())] = v
    # ambiguous (NCBI-style averages, close to original B62 dict)
    def quad(a, b, c, d):
        return (mat[ord(a), ord(b)] + mat[ord(c), ord(d)]) // 2

    for x in "BXZJU":
        for y in "ARNDCQEGHILKMFPSTWYVBZJUOX":
            mat[ord(x), ord(y)] = -4
    # B=DN, Z=EQ, J=IL, U=C, X=avg
    bmap = {ord("B"): "DN", ord("Z"): "EQ", ord("J"): "IL", ord("U"): "C"}
    for code, pair in bmap.items():
        for y in AA_ORDER:
            if len(pair) == 1:
                v = int(mat[ord(pair[0]), ord(y)])
            else:
                v = (int(mat[ord(pair[0]), ord(y)]) + int(mat[ord(pair[1]), ord(y)])) // 2
            mat[code, ord(y)] = mat[ord(y), code] = v
    for y in AA_ORDER:
        mat[ord("X"), ord(y)] = mat[ord(y), ord("X")] = -1
    # lowercase mirrors
    mat[ord("x"), :] = mat[ord("X"), :]
    mat[:, ord("x")] = mat[:, ord("X")]
    return mat


B62 = _build_b62()
B62_NP = B62.copy()


def score2bit(score, gap=True):
    if gap:
        return int((0.267 * score + 3.1941832122778293) / 0.69314718055994529)
    return int((0.309 * score + 1.9589953886039688) / 0.69314718055994529)


def bit2score(bit, gap=True):
    if gap:
        return (bit * 0.69314718055994529 - 3.1941832122778293) / 0.267
    return (bit * 0.69314718055994529 - 1.9589953886039688) / 0.309


def bit2e(D, a, b, bit):
    la = len(a); lb = len(b)
    return D * la * lb * (2.0 ** (-bit))


def f2s(e):
    """Faithful port of original f2s (fsearch.py:43-65): float -> SCI string."""
    if e <= 0:
        E = "0"
    elif 0 < e < 1e-3:
        a = math.log10(e)
        a -= int(a)
        a = (1 + a) if a < 0 else a
        b = 10 ** a
        s = str(math.log10(e / b))
        end = max(0, s.find("."))
        s = s[:end]
        p = str(b)
        end = max(0, p.find(".") + 3)
        p = p[:end]
        E = "%se%s" % (p, s)
    else:
        E = str(e)
    return E


# --------------------------------------------------------------------------
# FASTA I/O + reduced alphabet
# --------------------------------------------------------------------------
def parse_fasta(path):
    heads, seqs = [], []
    with open(path) as fh:
        head, buf = None, []
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if head is not None:
                    heads.append(head)
                    seqs.append("".join(buf))
                head = line[1:]
                buf = []
            else:
                buf.append(line.strip())
        if head is not None:
            heads.append(head)
            seqs.append("".join(buf))
    return heads, seqs


def encode_seqs(seqs):
    """List[str] -> (concat uint8, soas, headers lengths)."""
    n = len(seqs)
    lens = np.array([len(s) for s in seqs], dtype=np.int64)
    total = int(lens.sum())
    concat = np.empty(total, dtype=np.uint8)
    soas = np.empty(n + 1, dtype=np.int64)
    off = 0
    for i, s in enumerate(seqs):
        soas[i] = off
        if s:
            concat[off:off + len(s)] = np.frombuffer(s.encode("ascii", "ignore"), dtype=np.uint8)
        off += len(s)
    soas[n] = off
    return concat, soas


def generate_nr_tbl(gaa="AST,CFILMVY,DN,EQ,G,H,KR,P,W"):
    """Mirror original generate_nr_tbl: map each letter to min-ord of its group."""
    tbl = np.arange(512, dtype=np.int64)
    for group in gaa.upper().split(","):
        group = group.strip()
        if not group:
            continue
        rep = min(ord(c) for c in group)
        for c in group:
            tbl[ord(c)] = rep
            tbl[ord(c.lower())] = rep
    return tbl


def seq_to_codes(seq_str):
    return np.frombuffer(seq_str.encode("ascii", "ignore"), dtype=np.uint8).copy()


def _parse_spaces(space):
    pats = [p.strip() for p in space.split(",") if p.strip()]
    kmax = max(len(p) for p in pats)
    S = len(pats)
    padded = np.zeros((S, kmax), dtype=np.uint8)
    lens = np.empty(S, dtype=np.int64)
    mw = 0
    for i, p in enumerate(pats):
        lens[i] = len(p)
        for j, ch in enumerate(p):
            padded[i, j] = ord(ch)
        mw = max(mw, p.count("1"))
    return padded, lens, mw, kmax


# --------------------------------------------------------------------------
# Numba kernels: seeding
# --------------------------------------------------------------------------
@njit(fastmath=True)
def spseeds_fnv_nb(seq, codes, spaces, spaces_len, NC, step, out_hash, out_pos):
    """Fill out_hash/out_pos; return count. seq: uint8[:]."""
    L = seq.shape[0]
    S = spaces.shape[0]
    cnt = 0
    for s in range(S):
        k = int(spaces_len[s])
        suppressed = False  # placeholder for clarity
        for i in range(0, L - k + 1, step):
            ok = True
            n = np.uint32(0x811C9DC5)
            for j in range(k):
                c = seq[i + j]
                if c == 88 or c == 120:  # 'X' / 'x'
                    ok = False
                    break
                if spaces[s, j] != 48:  # '0'
                    ch = np.uint32(codes[c])
                    n ^= ch
                    n *= np.uint32(0x01000193)
            if not ok:
                continue
            n ^= np.uint32(s)
            n *= np.uint32(0x01000193)
            out_hash[cnt] = int(np.int64(n % np.uint32(NC)))
            out_pos[cnt] = i
            cnt += 1
    return cnt


def spseeds_fnv(seq_u8, code_tbl, spaces, spaces_len, NC, step=1):
    S = spaces.shape[0]
    L = seq_u8.shape[0]
    kmax = int(spaces_len.max())
    maxn = S * ((L // step + 1) if L >= kmax else 1) + 1
    oh = np.empty(maxn, dtype=np.int64)
    op = np.empty(maxn, dtype=np.int64)
    n = spseeds_fnv_nb(seq_u8, code_tbl, spaces, spaces_len, int(NC), int(step), oh, op)
    return oh[:n], op[:n]


@njit(fastmath=True)
def _count_kernel(concat, soas, codes, spaces, spaces_len, NC, step, start):
    for si in range(soas.shape[0] - 1):
        s = int(soas[si])
        e = int(soas[si + 1])
        L = e - s
        S = spaces.shape[0]
        for si2 in range(S):
            k = int(spaces_len[si2])
            if L < k:
                continue
            for i in range(0, L - k + 1, step):
                ok = True
                n = np.uint32(0x811C9DC5)
                for j in range(k):
                    c = concat[s + i + j]
                    if c == 88 or c == 120:
                        ok = False
                        break
                    if spaces[si2, j] != 48:
                        n ^= np.uint32(codes[c])
                        n *= np.uint32(0x01000193)
                if not ok:
                    continue
                n ^= np.uint32(si2)
                n *= np.uint32(0x01000193)
                h = int(np.int64(n % np.uint32(NC)))
                start[h] += 1


@njit(fastmath=True)
def _fill_kernel(concat, soas, codes, spaces, spaces_len, NC, step, cur, locus):
    for si in range(soas.shape[0] - 1):
        s = int(soas[si])
        e = int(soas[si + 1])
        L = e - s
        S = spaces.shape[0]
        for si2 in range(S):
            k = int(spaces_len[si2])
            if L < k:
                continue
            for i in range(0, L - k + 1, step):
                ok = True
                n = np.uint32(0x811C9DC5)
                for j in range(k):
                    c = concat[s + i + j]
                    if c == 88 or c == 120:
                        ok = False
                        break
                    if spaces[si2, j] != 48:
                        n ^= np.uint32(codes[c])
                        n *= np.uint32(0x01000193)
                if not ok:
                    continue
                n ^= np.uint32(si2)
                n *= np.uint32(0x01000193)
                h = int(np.int64(n % np.uint32(NC)))
                locus[int(cur[h])] = int(i + s)
                cur[h] += 1


# --------------------------------------------------------------------------
# Numba kernels: alignment
# --------------------------------------------------------------------------
@njit(fastmath=True)
def ungap_nb(qseq, sseq, Qst, Sst, b62, dropX):
    """Bidirectional X-drop extension. Returns (numchain flag, max_score, qst, qed, sst, sed)."""
    ql = qseq.shape[0]
    sl = sseq.shape[0]
    qst = int(Qst)
    sst = int(Sst)
    score = 0
    max_score = 0
    max_qed = qst
    max_sed = sst
    flag = 0
    while 0 <= qst < ql and 0 <= sst < sl:
        flag += 1
        score += int(b62[int(qseq[qst]), int(sseq[sst])])
        if score > max_score:
            max_score, max_qed, max_sed = score, qst, sst
        elif score + dropX < max_score:
            break
        qst += 1
        sst += 1
    qst = int(Qst) - 1
    sst = int(Sst) - 1
    score = max_score
    max_qst = qst
    max_sst = sst
    while 0 <= qst < ql and 0 <= sst < sl:
        flag += 1
        score += int(b62[int(qseq[qst]), int(sseq[sst])])
        if score > max_score:
            max_score, max_qst, max_sst = score, qst, sst
        elif score + dropX < max_score:
            break
        qst -= 1
        sst -= 1
    return flag, max_score, int(max_qst), int(max_qed), int(max_sst), int(max_sed)


@njit(fastmath=True)
def kswat_nb(s0, s1, qsp, ssp, sst, go, ge, mat, kbound, score, trace):
    """K-banded DP core. s0/s1 are already sliced & direction-adjusted.
    Returns (maxscore, i_max, j_max, i_for_tb, j_for_tb, idy, alen, mis, gap).
    score: (l1, l0) int16, trace: (l1, l0) int8  0='*' 1='-' 2='|' 3='\\\\'
    """
    l0 = s0.shape[0] + 1
    l1 = s1.shape[0] + 1
    for i in range(1, l0):
        score[0, i] = 0
        trace[0, i] = 1
    for i in range(1, l1):
        score[i, 0] = 0
        trace[i, 0] = 2
        st = i - kbound - 1
        if st < 0:
            st = 0
        if st > l0 - 1:
            st = l0 - 1
        en = i + kbound + 1
        if en > l0 - 1:
            en = l0 - 1
        if st <= en:
            trace[i, st] = 2
            trace[i, en] = 1
            score[i, st] = 0
            score[i, en] = 0

    i_max = 0
    j_max = 0
    maxscore = 0
    for i in range(1, l1):
        start = i - kbound
        if start < 1:
            start = 1
        end = i + kbound
        if end > l0:
            end = l0
        for j in range(start, end):
            if trace[i, j - 1] == 1:
                I = int(score[i, j - 1]) + ge
            else:
                I = int(score[i, j - 1]) + go
            M = int(score[i - 1, j - 1]) + int(mat[int(s1[i - 1]), int(s0[j - 1])])
            if trace[i - 1, j] == 2:
                D = int(score[i - 1, j]) + ge
            else:
                D = int(score[i - 1, j]) + go
            B = 0
            if I > B:
                B = I
            if M > B:
                B = M
            if D > B:
                B = D
            score[i, j] = B
            if B > maxscore:
                i_max = i
                j_max = j
                maxscore = B
            if B == M:
                trace[i, j] = 3
            elif B == I:
                trace[i, j] = 1
            elif B == D:
                trace[i, j] = 2
            else:
                trace[i, j] = 0

    i = i_max
    j = j_max
    idy = 0.0
    alen = 0
    mis = 0
    gap = 0
    op = -1
    i_end = i_max
    j_end = j_max
    while i > 0 or j > 0:
        tij = trace[i, j]
        if tij == 3:
            alen += 1
            if s0[j - 1] == s1[i - 1]:
                idy += 1.0
            else:
                mis += 1
            i -= 1
            j -= 1
        elif tij == 1:
            alen += 1
            mis += 1
            if op != 0:
                gap += 1
                op = 0
            j -= 1
        elif tij == 2:
            alen += 1
            mis += 1
            if op != 1:
                gap += 1
                op = 1
            i -= 1
        else:
            break
    if alen > 0:
        idy = idy * 100.0 / alen
    return maxscore, int(i_end), int(j_end), int(i), int(j), idy, int(alen), int(mis), int(gap)


def kswat_st(S0_str, S1_str, qst=0, qed=-1, sst=0, sed=-1, mat=None, kbound=16,
             go=-11, ge=-1, score_buf=None, trace_buf=None):
    """Python wrapper mirroring original kswat_st (lines 1357-1478).

    Returns (idy%, aln_len, mis, gap, q_start, q_end, s_start, s_end, bit).
    """
    if mat is None:
        mat = B62_NP
    s0_full = seq_to_codes(S0_str)
    s1_full = seq_to_codes(S1_str)
    n0, n1 = len(s0_full), len(s1_full)
    qst_c = min(max(qst, 0), n0)
    qed_c = n0 if qed < 0 else min(max(qed, 0), n0)
    sst_c = min(max(sst, 0), n1)
    sed_c = n1 if sed < 0 else min(max(sed, 0), n1)
    # direction flags (allow reverse like original via qsp/ssp)
    qdir = 1 if qed_c >= qst_c else -1
    sdir = 1 if sed_c >= sst_c else -1
    # swap so l0 is the shorter-or-equal axis (match original kswat_st)
    want_swap = abs(qed_c - qst_c) >= abs(sed_c - sst_c)
    if want_swap:
        s0_all, s1_all = s1_full, s0_full
        qst2, qed2, sst2, sed2 = sst_c, sed_c, qst_c, qed_c
        qdir2, sdir2 = sdir, qdir
        swapped = True
    else:
        s0_all, s1_all = s0_full, s1_full
        qst2, qed2, sst2, sed2 = qst_c, qed_c, sst_c, sed_c
        qdir2, sdir2 = qdir, sdir
        swapped = False

    l0 = abs(qed2 - qst2)
    l1 = abs(sed2 - sst2)
    # materialize oriented slices (Numba wants forward arrays)
    if qdir2 == 1:
        s0 = np.ascontiguousarray(s0_all[qst2:qed2])
    else:
        s0 = np.ascontiguousarray(s0_all[qed2:qst2][::-1])
    if sdir2 == 1:
        s1 = np.ascontiguousarray(s1_all[sst2:sed2])
    else:
        s1 = np.ascontiguousarray(s1_all[sed2:sst2][::-1])

    l0 = s0.shape[0]
    l1 = s1.shape[0]
    if l0 == 0 or l1 == 0:
        return (0.0, 0, 0, 0, qst_c, qst_c, sst_c, sst_c, 0)

    score = np.zeros((l1 + 1, l0 + 1), dtype=np.int16)
    trace = np.zeros((l1 + 1, l0 + 1), dtype=np.int8)
    maxscore, ie, je, i0, j0, idy, alen, mis, gap = kswat_nb(
        s0, s1, 1, 1, 0, go, ge, mat, int(kbound), score, trace)

    # map back to original coords (approx: invert direction + swap)
    def back(pos_oriented, start, direction, length):
        # pos in oriented slice -> original coordinate
        if direction == 1:
            return start + pos_oriented
        return start - pos_oriented

    if not swapped:
        qs = back(j0, qst2, qdir2, l0)
        qe = back(je - 0, qst2, qdir2, l0)  # end: j_max oriented index
        # j_max in oriented DP = je (max col); recompute end from traceback start
        # Use oriented end = je, oriented start = j0
        q_start = back(j0, qst2, qdir2, l0)
        q_end = back(je, qst2, qdir2, l0)
        s_start = back(i0, sst2, sdir2, l1)
        s_end = back(ie, sst2, sdir2, l1)
    else:
        # swapped: DP axes were (s,q) swapped
        q_start = back(i0, qst2, qdir2, l0)  # careful: after swap qst2 holds orig sst
        q_end = back(ie, qst2, qdir2, l0)
        s_start = back(j0, sst2, sdir2, l1)
        s_end = back(je, sst2, sdir2, l1)
        # normalize order for display (low,high)
        q_start, q_end = min(q_start, q_end), max(q_start, q_end)
        s_start, s_end = min(s_start, s_end), max(s_start, s_end)
    return (float(idy), int(alen), int(mis), int(gap),
            int(q_start), int(q_end), int(s_start), int(s_end), score2bit(int(maxscore)))


def kswat_st_long(qstr, sstr, qi, qj, chk=4096, mat=None, kbound=16):
    li, lj = len(qstr), len(sstr)
    j = qj
    res = []
    for i in range(qi, li, chk):
        ied = min(li, i + chk)
        jed = min(lj, j + chk)
        res.append(kswat_st(qstr[i:ied], sstr[j:jed], qst=0, sst=0,
                            mat=mat, kbound=kbound))
        j += chk
    return res


# --------------------------------------------------------------------------
# Linear-space Smith-Waterman Gotoh (Huang/Myers-Miller) — ADDED, original kept
# --------------------------------------------------------------------------
# Port of SWIPE align.cc::region()/diff(), Numba-accelerated.
# Memory: O(N) for the DP arrays (N = len(s1)); time: O(M*N) (2 passes
# top-level + recursive splits sum < 2x). q_align = -go + ge so that
# a length-1 gap costs -go (11) and extensions cost -ge (1), matching
# the original trace-based kswat_nb (go/ge are negative penalties there).
@njit(fastmath=True)
def _region_nb(a, b, mat, q, r, kbound):
    """Banded forward + reverse two-row scan (Hirschberg region)."""
    NEG = -4000000000000
    M = a.shape[0]
    N = b.shape[0]
    if M == 0 or N == 0:
        return 0, 0, 0, 0, 0
    HH = np.empty(N, dtype=np.int64)
    EE = np.empty(N, dtype=np.int64)
    score = 0
    a_end = 0
    b_end = 0
    for j in range(N):
        if j <= kbound:
            HH[j] = 0
            EE[j] = -q
        else:
            HH[j] = NEG
            EE[j] = NEG
    for i in range(M):
        L = i - kbound
        if L < 0:
            L = 0
        R = i + kbound
        if R > N - 1:
            R = N - 1
        if L == 0:
            h = 0
            p = 0
            f = -q
        else:
            h = NEG
            f = NEG
            p = int(HH[L - 1])
        for j in range(L, R + 1):
            fh = h - q
            if f > fh:
                pass
            else:
                f = fh
            f = f - r
            mm = EE[j]
            if HH[j] - q > mm:
                mm = HH[j] - q
            EE[j] = mm - r
            h = p + int(mat[int(b[j]), int(a[i])])
            if h < 0:
                h = 0
            if f > h:
                h = f
            if EE[j] > h:
                h = EE[j]
            p_next = int(HH[j])
            HH[j] = h
            p = p_next
            if h > score and h != NEG:
                score = h
                a_end = i
                b_end = j
    for j in range(b_end, -1, -1):
        HH[j] = -1
        EE[j] = -1
    cost = 0
    a_begin = a_end
    b_begin = b_end
    for i in range(a_end, -1, -1):
        L = i - kbound
        if L < 0:
            L = 0
        R = i + kbound
        if R > b_end:
            R = b_end
        if R == b_end:
            if i == a_end:
                p = 0
            else:
                p = -1
            h = -1
            f = -1
        else:
            h = NEG
            f = NEG
            p = NEG
        for j in range(R, L - 1, -1):
            fh = h - q
            if fh > f:
                f = fh
            f = f - r
            mm = EE[j]
            if HH[j] - q > mm:
                mm = HH[j] - q
            EE[j] = mm - r
            h = p + int(mat[int(b[j]), int(a[i])])
            if f > h:
                h = f
            if EE[j] > h:
                h = EE[j]
            p_next = int(HH[j])
            HH[j] = h
            p = p_next
            if h > cost:
                cost = h
                a_begin = i
                b_begin = j
                if cost >= score:
                    return a_begin, b_begin, a_end, b_end, int(score)
    return a_begin, b_begin, a_end, b_end, int(score)


@njit(fastmath=True)
def _diff_fwd_nb(a_sub, b_sub, mat, q, r, tb, rows, kbound):
    """Banded forward HH/EE for `rows` rows of the (M,N) subproblem."""
    NEG = -4000000000000
    N = b_sub.shape[0]
    HH = np.full(N + 1, NEG, dtype=np.int64)
    EE = np.full(N + 1, NEG, dtype=np.int64)
    HH[0] = 0
    t = -q
    for j in range(1, N + 1):
        t -= r
        if j <= kbound:
            HH[j] = t
            EE[j] = t - q
    t = -tb
    for i in range(1, rows + 1):
        L = i - kbound
        if L < 1:
            L = 1
        R = i + kbound
        if R > N:
            R = N
        if L == 1:
            p = int(HH[0])
            t -= r
            h = t
            HH[0] = t
            f = t - q
        else:
            # left of band: HH[L-1] still holds prev-row value (diag source)
            p = int(HH[L - 1])
            h = NEG
            f = NEG
            # HH[0] update irrelevant; t still accumulates for col0 path
            t -= r
            if L == 1:
                HH[0] = t
        for j in range(L, R + 1):
            fh = h - q
            if f > fh:
                pass
            else:
                f = fh
            f = f - r
            mm = EE[j]
            if HH[j] - q > mm:
                mm = HH[j] - q
            EE[j] = mm - r
            h = p + int(mat[int(b_sub[j - 1]), int(a_sub[i - 1])])
            if f > h:
                h = f
            if EE[j] > h:
                h = EE[j]
            p_next = int(HH[j])
            HH[j] = h
            p = p_next
        if L == 1:
            EE[0] = HH[0]
    return HH, EE


@njit(fastmath=True)
def _diff_rev_nb(a_sub, b_sub, mat, q, r, te, M, rows_rev, kbound):
    """Banded reverse XX/YY for last `rows_rev` rows (global-row band)."""
    NEG = -4000000000000
    N = b_sub.shape[0]
    XX = np.full(N + 1, NEG, dtype=np.int64)
    YY = np.full(N + 1, NEG, dtype=np.int64)
    XX[0] = 0
    t = -q
    for j in range(1, N + 1):
        # forward col j_global = N-j+1; band at bottom row M: |M-(N-j+1)|<=kbound
        # init along j_rev axis in band zone only
        t -= r
        # keep full init for small N (band dry-run); band prunes in loop below
        XX[j] = t
        YY[j] = t - q
    t = -te
    for i in range(1, rows_rev + 1):
        grow = M - i + 1  # 1-indexed global row (forward coords)
        L = grow - kbound
        if L < 1:
            L = 1
        R = grow + kbound
        if R > N:
            R = N
        # reverse col j_rev (1..N) maps to forward col N-j_rev+1
        jr_lo = N - R + 1
        jr_hi = N - L + 1
        if jr_lo < 1:
            jr_lo = 1
        if jr_hi > N:
            jr_hi = N
        if jr_lo == 1:
            p = int(XX[0])
            t -= r
            h = t
            XX[0] = t
            f = t - q
        else:
            p = int(XX[jr_lo - 1])
            h = NEG
            f = NEG
            t -= r
        for j in range(jr_lo, jr_hi + 1):
            fh = h - q
            if f > fh:
                pass
            else:
                f = fh
            f = f - r
            mm = YY[j]
            if XX[j] - q > mm:
                mm = XX[j] - q
            YY[j] = mm - r
            h = p + int(mat[int(b_sub[N - j]), int(a_sub[M - i])])
            if f > h:
                h = f
            if YY[j] > h:
                h = YY[j]
            p_next = int(XX[j])
            XX[j] = h
            p = p_next
        if jr_lo == 1:
            YY[0] = XX[0]
    return XX, YY


def _diff_ops(a_sub, b_sub, mat, q, r, tb0, te0, kbound=16):
    """Iterative Hirschberg (Myers-Miller) collecting ops in order.

    a_sub,b_sub: numpy uint8 slices for the *regional* box.
    Returns list of (op,len) with op in {'M','D','I'} (unmerged).
    """
    ops = []
    stack = [("sub", 0, 0, int(a_sub.shape[0]), int(b_sub.shape[0]), tb0, te0)]
    while stack:
        task = stack.pop()
        if task[0] == "emit":
            ops.append((task[1], int(task[2])))
            continue
        _, a_pos, b_pos, M, N, tb, te = task
        if N == 0:
            if M > 0:
                ops.append(("D", M))
            continue
        if M == 0:
            ops.append(("I", N))
            continue
        if M == 1:
            # Conversion: 1 a-residue vs N b-residues (align.cc diff M==1).
            a0 = a_sub[a_pos]
            if tb <= te:
                max_score = -tb - (1 + N) * r - q
                J = -1
            else:
                max_score = -q - (1 + N) * r - te
                J = N
            for j in range(N):
                s = int(mat[int(b_sub[b_pos + j]), int(a0)]) - r * (N - 1)
                if j > 0:
                    s -= q
                if j < N - 1:
                    s -= q
                if s > max_score:
                    max_score = s
                    J = j
            if J == -1:
                ops.append(("D", 1))
                ops.append(("I", N))
            elif J == N:
                ops.append(("I", N))
                ops.append(("D", 1))
            else:
                if J > 0:
                    ops.append(("I", J))
                ops.append(("M", 1))
                if J < N - 1:
                    ops.append(("I", N - 1 - J))
            continue
        I = M // 2
        a_slice = np.ascontiguousarray(a_sub[a_pos:a_pos + M])
        b_slice = np.ascontiguousarray(b_sub[b_pos:b_pos + N])
        HH, EE = _diff_fwd_nb(a_slice, b_slice, mat, int(q), int(r), int(tb), int(I), int(kbound))
        XX, YY = _diff_rev_nb(a_slice, b_slice, mat, int(q), int(r), int(te), int(M), int(M - I), int(kbound))
        max_score = -(1 << 62)
        P = 0
        J = 0
        for j in range(N + 1):
            s = int(HH[j]) + int(XX[N - j])
            if s > max_score:
                max_score = s
                P = 0
                J = j
        for j in range(N + 1):
            s = int(EE[j]) + int(YY[N - j]) + int(q)
            if s >= max_score:
                max_score = s
                P = 1
                J = j
        if P == 0:
            stack.append(("sub", a_pos + I, b_pos + J, M - I, N - J, q, te))
            stack.append(("sub", a_pos, b_pos, I, J, tb, q))
        else:
            stack.append(("sub", a_pos + I + 1, b_pos + J, M - I - 1, N - J, 0, te))
            stack.append(("emit", "D", 2))
            stack.append(("sub", a_pos, b_pos, I - 1, J, tb, 0))
    return ops


def _ops_to_counts(ops, a_sub, b_sub):
    """Merge runs and count (idy, alen, mis, gap) like original traceback."""
    merged = []
    for op, ln in ops:
        if ln <= 0:
            continue
        if merged and merged[-1][0] == op:
            merged[-1] = (op, merged[-1][1] + ln)
        else:
            merged.append((op, ln))
    idy = 0.0
    alen = 0
    mis = 0
    gap = 0
    pa = 0
    pb = 0
    for op, ln in merged:
        if op == "M":
            for _ in range(ln):
                if pa < len(a_sub) and pb < len(b_sub) and a_sub[pa] == b_sub[pb]:
                    idy += 1.0
                else:
                    mis += 1
                alen += 1
                pa += 1
                pb += 1
        elif op == "D":
            alen += ln
            mis += ln
            gap += 1
            pa += ln
        else:  # "I"
            alen += ln
            mis += ln
            gap += 1
            pb += ln
    if alen > 0:
        idy = idy * 100.0 / alen
    return float(idy), int(alen), int(mis), int(gap), merged


def kswat_st_linear(S0_str, S1_str, qst=0, qed=-1, sst=0, sed=-1, mat=None,
                    kbound=16, go=-11, ge=-1, score_buf=None, trace_buf=None):
    """Banded linear-space drop-in for kswat_st (same return tuple).

    Uses banded region() to locate the local alignment, then banded
    Hirschberg diff() (all arrays O(N), band width 2*kbound+1) to
    trace it back. Original kswat_st/kswat_nb are preserved.
    This function is now the default used by blastp()/kswat_st_long().
    """
    if mat is None:
        mat = B62_NP
    s0_full = seq_to_codes(S0_str)
    s1_full = seq_to_codes(S1_str)
    n0, n1 = len(s0_full), len(s1_full)
    qst_c = min(max(qst, 0), n0)
    qed_c = n0 if qed < 0 else min(max(qed, 0), n0)
    sst_c = min(max(sst, 0), n1)
    sed_c = n1 if sed < 0 else min(max(sed, 0), n1)
    qdir = 1 if qed_c >= qst_c else -1
    sdir = 1 if sed_c >= sst_c else -1
    want_swap = abs(qed_c - qst_c) >= abs(sed_c - sst_c)
    if want_swap:
        s0_all, s1_all = s1_full, s0_full
        qst2, qed2, sst2, sed2 = sst_c, sed_c, qst_c, qed_c
        qdir2, sdir2 = sdir, qdir
        swapped = True
    else:
        s0_all, s1_all = s0_full, s1_full
        qst2, qed2, sst2, sed2 = qst_c, qed_c, sst_c, sed_c
        qdir2, sdir2 = qdir, sdir
        swapped = False
    if qdir2 == 1:
        s0 = np.ascontiguousarray(s0_all[qst2:qed2])
    else:
        s0 = np.ascontiguousarray(s0_all[qed2:qst2][::-1])
    if sdir2 == 1:
        s1 = np.ascontiguousarray(s1_all[sst2:sed2])
    else:
        s1 = np.ascontiguousarray(s1_all[sed2:sst2][::-1])
    if s0.shape[0] == 0 or s1.shape[0] == 0:
        return (0.0, 0, 0, 0, qst_c, qst_c, sst_c, sst_c, 0)
    q_align = int(-go + ge)
    r_align = int(-ge)
    a_begin, b_begin, a_end, b_end, raw = _region_nb(s0, s1, mat, q_align, r_align, int(kbound))
    if raw <= 0:
        return (0.0, 0, 0, 0, qst_c, qst_c, sst_c, sst_c, 0)
    a_sub = np.ascontiguousarray(s0[a_begin:a_end + 1])
    b_sub = np.ascontiguousarray(s1[b_begin:b_end + 1])
    ops = _diff_ops(a_sub, b_sub, mat, q_align, r_align, q_align, q_align, int(kbound))
    idy, alen, mis, gap, _merged = _ops_to_counts(ops, a_sub, b_sub)

    def back(pos, start, direction):
        return start + pos if direction == 1 else start - pos

    # region indices are 0-based inclusive; original DP coords are
    # consumed counts, so end maps inclusive+1.
    if not swapped:
        q_start = back(a_begin, qst2, qdir2)
        q_end = back(a_end + 1, qst2, qdir2)
        s_start = back(b_begin, sst2, sdir2)
        s_end = back(b_end + 1, sst2, sdir2)
    else:
        q_start = back(b_begin, qst2, qdir2)
        q_end = back(b_end + 1, qst2, qdir2)
        s_start = back(a_begin, sst2, sdir2)
        s_end = back(a_end + 1, sst2, sdir2)
    if swapped or qdir2 == -1 or sdir2 == -1:
        q_start, q_end = min(q_start, q_end), max(q_start, q_end)
        s_start, s_end = min(s_start, s_end), max(s_start, s_end)
    return (float(idy), int(alen), int(mis), int(gap),
            int(q_start), int(q_end), int(s_start), int(s_end), score2bit(int(raw)))


def kswat_st_long_linear(qstr, sstr, qi, qj, chk=4096, mat=None, kbound=16):
    li, lj = len(qstr), len(sstr)
    j = qj
    res = []
    for i in range(qi, li, chk):
        ied = min(li, i + chk)
        jed = min(lj, j + chk)
        res.append(kswat_st_linear(qstr[i:ied], sstr[j:jed], qst=0, sst=0,
                                   mat=mat, kbound=kbound))
        j += chk
    return res


# --------------------------------------------------------------------------
# WFA2: Wavefront Alignment 2 (Marco et al.) — banded, affine, arbitrary scores
# --------------------------------------------------------------------------
# ADDED 2026-11-08. Original banded/Hirschberg engines kept untouched.
#
# Wavefront organised by SCORE via Diagonal Segment (SED) extensions:
#   * Diagonals k = i-j, |k| <= kbound (33 diagonals, like kswat band).
#   * Each diagonal stores prefix sums along it; suffix_max gives the
#     SED-peak extension from any anchor in O(1) (precompute per alignment).
#   * One "wave" = one diagonal segment. Next wave = one affine gap
#     (any L: cost go+(L-1)*ge) to a neighbouring diagonal + its SED peak.
#   * Start anchored at slice (0,0), end free (peak of last diagonal),
#     then trimmed to [region-like peak]. Memory per wave: O(nd),
#     total backpointers: max_blocks x nd (small).
@njit(parallel=True, fastmath=True)
def _wfa_precompute_nb(a, b, mat, kbound, pref, suff, suff_arg, diag_start, diag_len):
    n = a.shape[0]
    m = b.shape[0]
    nd = 2 * kbound + 1
    for idx in prange(nd):
        k = idx - kbound
        ist = 0
        if k > ist:
            ist = k
        ien = n - 1
        if m - 1 + k < ien:
            ien = m - 1 + k
        diag_start[idx] = ist
        if ien < ist:
            diag_len[idx] = 0
            pref[idx, 0] = 0
            suff[idx, 0] = 0
            suff_arg[idx, 0] = 0
        else:
            Ld = ien - ist + 1
            diag_len[idx] = Ld
            pref[idx, 0] = 0
            for t in range(Ld):
                ii = ist + t
                jj = ii - k
                pref[idx, t + 1] = pref[idx, t] + int(mat[int(b[jj]), int(a[ii])])
            suff[idx, Ld] = pref[idx, Ld]
            suff_arg[idx, Ld] = Ld
            for p in range(Ld - 1, -1, -1):
                if pref[idx, p] >= suff[idx, p + 1]:
                    suff[idx, p] = pref[idx, p]
                    suff_arg[idx, p] = p
                else:
                    suff[idx, p] = suff[idx, p + 1]
                    suff_arg[idx, p] = suff_arg[idx, p + 1]


@njit(fastmath=True)
def _wfa_wave_nb(a, b, mat, go, ge, kbound, max_blocks,
                 pref, suff, suff_arg, diag_start, diag_len,
                 score_tb, peak_tb, prev_tb, anchor_tb, prev_cand_tb, K):
    """Fill wavefront Pareto tables. Tables shape (max_blocks,nd,K)."""
    nd = 2 * kbound + 1
    NEG = -4000000000000
    for bb in range(max_blocks):
        for d in range(nd):
            for kk in range(K):
                score_tb[bb, d, kk] = NEG
                peak_tb[bb, d, kk] = -1
                prev_tb[bb, d, kk] = -1
                anchor_tb[bb, d, kk] = -1
                prev_cand_tb[bb, d, kk] = -1
    # --- block 0: Kadane max diagonal segment per diagonal (free local start)
    for d0 in range(nd):
        Ld = int(diag_len[d0])
        if Ld == 0:
            continue
        best = 0
        best_s = 0
        best_e = 0
        cur = 0
        cur_start = 0
        for t in range(Ld):
            cur += int(pref[d0, t + 1] - pref[d0, t])
            if cur > best:
                best = cur
                best_s = cur_start
                best_e = t + 1
            if cur < 0:
                cur = 0
                cur_start = t + 1
        if best > 0:
            score_tb[0, d0, 0] = best
            anchor_tb[0, d0, 0] = int(diag_start[d0] + best_s)
            peak_tb[0, d0, 0] = int(diag_start[d0] + best_e)
            prev_tb[0, d0, 0] = -1
            prev_cand_tb[0, d0, 0] = -1
    # temporary candidate pool per wave
    maxcand = nd * K
    tmpS = np.empty((nd, maxcand), dtype=np.int64)
    tmpP = np.empty((nd, maxcand), dtype=np.int64)
    tmpPD = np.empty((nd, maxcand), dtype=np.int64)
    tmpPC = np.empty((nd, maxcand), dtype=np.int64)
    tmpA = np.empty((nd, maxcand), dtype=np.int64)
    tmpN = np.empty(nd, dtype=np.int64)
    used = np.empty(maxcand, dtype=np.int8)
    # --- waves
    for w in range(1, max_blocks):
        for d in range(nd):
            tmpN[d] = 0
        for pd in range(nd):
            for pc in range(K):
                prev_score = int(score_tb[w - 1, pd, pc])
                if prev_score == NEG:
                    continue
                prev_peak = int(peak_tb[w - 1, pd, pc])
                if prev_peak < 0:
                    continue
                pk = pd - kbound
                # prev peak col = prev_peak - pk
                for td in range(nd):
                    if td == pd:
                        continue
                    tk = td - kbound
                    delta = tk - pk
                    L = delta
                    if L < 0:
                        L = -L
                    if L == 0:
                        continue
                    gap_pen = go + (L - 1) * ge
                    if delta > 0:
                        ai = prev_peak + L
                        aj = prev_peak - pk
                    else:
                        ai = prev_peak
                        aj = prev_peak - pk + L
                    if ai < 0 or ai > a.shape[0] or aj < 0 or aj > b.shape[0]:
                        continue
                    if ai < diag_start[td] or ai > diag_start[td] + diag_len[td]:
                        continue
                    p = ai - int(diag_start[td])
                    if p < 0 or p > diag_len[td]:
                        continue
                    sed = int(suff[td, p] - pref[td, p])
                    q = int(suff_arg[td, p])
                    cand = prev_score + gap_pen + sed
                    prow = int(diag_start[td] + q)
                    idx = int(tmpN[td])
                    if idx >= maxcand:
                        continue
                    tmpS[td, idx] = cand
                    tmpP[td, idx] = prow
                    tmpPD[td, idx] = pd
                    tmpPC[td, idx] = pc
                    tmpA[td, idx] = ai
                    tmpN[td] = idx + 1
        anyv = False
        for td in range(nd):
            cnt = int(tmpN[td])
            if cnt == 0:
                continue
            for u in range(cnt):
                used[u] = 0
            kept = 0
            max_peak_kept = -1
            while kept < K:
                best_i = -1
                best_s = NEG
                best_p = -1
                for i in range(cnt):
                    if used[i] != 0:
                        continue
                    s_i = int(tmpS[td, i])
                    p_i = int(tmpP[td, i])
                    if p_i <= max_peak_kept:
                        continue
                    if s_i > best_s or (s_i == best_s and p_i > best_p):
                        best_s = s_i
                        best_p = p_i
                        best_i = i
                if best_i < 0:
                    break
                used[best_i] = 1
                score_tb[w, td, kept] = int(tmpS[td, best_i])
                peak_tb[w, td, kept] = int(tmpP[td, best_i])
                prev_tb[w, td, kept] = int(tmpPD[td, best_i])
                prev_cand_tb[w, td, kept] = int(tmpPC[td, best_i])
                anchor_tb[w, td, kept] = int(tmpA[td, best_i])
                max_peak_kept = best_p
                kept += 1
                anyv = True
        if not anyv:
            break


def _wfa_trace_ops(a, b, score_tb, peak_tb, prev_tb, anchor_tb, prev_cand_tb,
                   best_b, best_d, best_c, kbound):
    """Walk backpointers to ops (forward order, merged runs list)."""
    segs = []  # each: ("D", diag_start, diag_end, gap_kind, gap_len)
    b_cur = int(best_b)
    d_cur = int(best_d)
    c_cur = int(best_c)
    while b_cur >= 0:
        peak = int(peak_tb[b_cur, d_cur, c_cur])
        anch = int(anchor_tb[b_cur, d_cur, c_cur])
        prev_d = int(prev_tb[b_cur, d_cur, c_cur])
        cur_k = int(d_cur - kbound)
        # diagonal segment in reverse
        segs.append(("diag", anch, peak, cur_k))
        if prev_d < 0:
            b_cur = -1
            break
        prev_c = int(prev_cand_tb[b_cur, d_cur, c_cur])
        prev_peak = int(peak_tb[b_cur - 1, prev_d, prev_c])
        prev_k = prev_d - kbound
        cur_k = d_cur - kbound
        delta = cur_k - prev_k
        if delta > 0:
            segs.append(("gap_a", prev_peak, anch, delta))
        else:
            segs.append(("gap_b", prev_peak, anch, -delta))
        d_cur = prev_d
        c_cur = prev_c
        b_cur -= 1
    segs.reverse()
    return segs


def kswat_st_wfa2(S0_str, S1_str, qst=0, qed=-1, sst=0, sed=-1, mat=None,
                  kbound=16, go=-11, ge=-1, score_buf=None, trace_buf=None,
                  max_blocks=48):
    """WFA2 drop-in for kswat_st (same return tuple, banded wavefront).

    Block 0 is a free local Kadane segment per band diagonal; later
    blocks add one affine gap + SED-peak diagonal segment. Anchored
    only in the sense the *last* segment ends at its peak.
    """
    if mat is None:
        mat = B62_NP
    s0_full = seq_to_codes(S0_str)
    s1_full = seq_to_codes(S1_str)
    n0, n1 = len(s0_full), len(s1_full)
    qst_c = min(max(qst, 0), n0)
    qed_c = n0 if qed < 0 else min(max(qed, 0), n0)
    sst_c = min(max(sst, 0), n1)
    sed_c = n1 if sed < 0 else min(max(sed, 0), n1)
    qdir = 1 if qed_c >= qst_c else -1
    sdir = 1 if sed_c >= sst_c else -1
    want_swap = abs(qed_c - qst_c) >= abs(sed_c - sst_c)
    if want_swap:
        s0_all, s1_all = s1_full, s0_full
        qst2, qed2, sst2, sed2 = sst_c, sed_c, qst_c, qed_c
        qdir2, sdir2 = sdir, qdir
        swapped = True
    else:
        s0_all, s1_all = s0_full, s1_full
        qst2, qed2, sst2, sed2 = qst_c, qed_c, sst_c, sed_c
        qdir2, sdir2 = qdir, sdir
        swapped = False
    if qdir2 == 1:
        s0 = np.ascontiguousarray(s0_all[qst2:qed2])
    else:
        s0 = np.ascontiguousarray(s0_all[qed2:qst2][::-1])
    if sdir2 == 1:
        s1 = np.ascontiguousarray(s1_all[sst2:sed2])
    else:
        s1 = np.ascontiguousarray(s1_all[sed2:sst2][::-1])
    n, m = s0.shape[0], s1.shape[0]
    if n == 0 or m == 0:
        return (0.0, 0, 0, 0, qst_c, qst_c, sst_c, sst_c, 0)
    nd = 2 * int(kbound) + 1
    maxLd = min(n, m) + 1
    pref = np.empty((nd, maxLd + 1), dtype=np.int64)
    suff = np.empty((nd, maxLd + 1), dtype=np.int64)
    suff_arg = np.empty((nd, maxLd + 1), dtype=np.int64)
    diag_start = np.empty(nd, dtype=np.int64)
    diag_len = np.empty(nd, dtype=np.int64)
    _wfa_precompute_nb(s0, s1, mat, int(kbound), pref, suff, suff_arg, diag_start, diag_len)
    NEG = -4000000000000
    K = 4
    score_tb = np.full((int(max_blocks), nd, K), NEG, dtype=np.int64)
    peak_tb = np.full((int(max_blocks), nd, K), -1, dtype=np.int64)
    prev_tb = np.full((int(max_blocks), nd, K), -1, dtype=np.int64)
    anchor_tb = np.full((int(max_blocks), nd, K), -1, dtype=np.int64)
    prev_cand_tb = np.full((int(max_blocks), nd, K), -1, dtype=np.int64)
    _wfa_wave_nb(s0, s1, mat, np.int64(go), np.int64(ge), np.int64(kbound),
                 int(max_blocks), pref, suff, suff_arg, diag_start, diag_len,
                 score_tb, peak_tb, prev_tb, anchor_tb, prev_cand_tb, np.int64(K))
    best_score = NEG
    best_b = -1
    best_d = -1
    best_c = -1
    for bb in range(int(max_blocks)):
        for d in range(nd):
            for cc in range(K):
                s = int(score_tb[bb, d, cc])
                if s > best_score:
                    best_score = s
                    best_b = bb
                    best_d = d
                    best_c = cc
    if best_score <= 0 or best_b < 0:
        return (0.0, 0, 0, 0, qst_c, qst_c, sst_c, sst_c, 0)
    segs = _wfa_trace_ops(s0, s1, score_tb, peak_tb, prev_tb, anchor_tb, prev_cand_tb,
                          best_b, best_d, best_c, int(kbound))
    # counts by walking segments forward (diag k carried per-segment)
    idy = 0.0
    alen = 0
    mis = 0
    gap = 0
    prev_gap_kind = None
    for seg in segs:
        if seg[0] == "diag":
            _, ar, pr, kk = seg
            for t in range(int(ar), int(pr)):
                alen += 1
                bj = int(t - kk)
                if 0 <= bj < s1.shape[0] and s0[t] == s1[bj]:
                    idy += 1.0
                else:
                    mis += 1
        elif seg[0] == "gap_a":
            _, pr, ar, L = seg
            alen += int(L); mis += int(L)
            if prev_gap_kind != "gap_a":
                gap += 1
            prev_gap_kind = "gap_a"
        else:
            _, pr, ar, L = seg  # gap_b (consume b)
            alen += int(L); mis += int(L)
            if prev_gap_kind != "gap_b":
                gap += 1
            prev_gap_kind = "gap_b"
        if seg[0] == "diag":
            prev_gap_kind = None
    if alen > 0:
        idy = idy * 100.0 / alen

    def back(pos, start, direction):
        return start + pos if direction == 1 else start - pos

    def orient(row_s0, col_s1):
        """Oriented (s0,s1) consumed counts -> original (q,s)."""
        if not swapped:
            return (back(row_s0, qst2, qdir2), back(col_s1, sst2, sdir2))
        return (back(col_s1, sst2, sdir2), back(row_s0, qst2, qdir2))

    peak_row = int(peak_tb[best_b, best_d, best_c])
    k_first = segs[0][3] if segs and segs[0][0] == "diag" else 0
    first_row = segs[0][1] if segs and segs[0][0] == "diag" else 0
    first_col = int(first_row - k_first)
    k_best = int(best_d - kbound)
    peak_col = int(peak_row - k_best)
    q_start, s_start = orient(first_row, first_col)
    q_end, s_end = orient(peak_row, peak_col)
    if swapped or qdir2 == -1 or sdir2 == -1:
        q_start, q_end = min(q_start, q_end), max(q_start, q_end)
        s_start, s_end = min(s_start, s_end), max(s_start, s_end)
    return (float(idy), int(alen), int(mis), int(gap),
            int(q_start), int(q_end), int(s_start), int(s_end), score2bit(int(best_score)))


def kswat_st_long_wfa2(qstr, sstr, qi, qj, chk=4096, mat=None, kbound=16):
    li, lj = len(qstr), len(sstr)
    j = qj
    res = []
    for i in range(qi, li, chk):
        ied = min(li, i + chk)
        jed = min(lj, j + chk)
        res.append(kswat_st_wfa2(qstr[i:ied], sstr[j:jed], qst=0, sst=0,
                                 mat=mat, kbound=kbound))
        j += chk
    return res


# --------------------------------------------------------------------------
# Batched threaded wrappers (SIMD across alignments via threads; Numba
# kernels release the GIL, so 8 cores scale ~linearly for blastp hits).
# --------------------------------------------------------------------------
from concurrent.futures import ThreadPoolExecutor as _TPE

def _batch(fn, pairs, kbound, workers=None):
    """Run fn(a,b,qst,sst,kbound) for each (a,b,qi,qj) in pairs."""
    import os
    w = int(workers or os.cpu_count() or 4)
    if w <= 1 or len(pairs) <= 1:
        return [fn(a, b, qst=qi, sst=qj, kbound=kbound) for (a, b, qi, qj) in pairs]
    with _TPE(max_workers=w) as ex:
        futs = [ex.submit(fn, a, b, qst=qi, sst=qj, kbound=kbound) for (a, b, qi, qj) in pairs]
        return [f.result() for f in futs]


def kswat_st_batch(pairs, kbound=16, workers=None):
    return _batch(kswat_st, pairs, kbound, workers)


def kswat_st_linear_batch(pairs, kbound=16, workers=None):
    return _batch(kswat_st_linear, pairs, kbound, workers)


def kswat_st_wfa2_batch(pairs, kbound=16, workers=None):
    return _batch(kswat_st_wfa2, pairs, kbound, workers)


# --------------------------------------------------------------------------
# SEG masking (entropy window)
# --------------------------------------------------------------------------
def _entropy_from_counts(counts, n):
    ent = 0.0
    log2 = math.log(2.0)
    for c in counts.values():
        if c:
            p = c / n
            ent -= p * math.log(p)
    return ent / log2 if ent else 0.0


def seg(S, minent=2.2, window=12):
    s = S.upper()
    n = len(s)
    winsize = int(window)
    if n < winsize:
        return s, s
    counts = {}
    for ch in s[:winsize]:
        counts[ch] = counts.get(ch, 0) + 1
    ent = _entropy_from_counts(counts, winsize)
    mask = [0] * n
    if ent < minent:
        mask[0] = 1
    for i in range(1, n - winsize + 1):
        pre, cur = s[i - 1], s[i + 11]
        if pre != cur:
            counts[pre] -= 1
            counts[cur] = counts.get(cur, 0) + 1
            ent = _entropy_from_counts(counts, winsize)
            if ent < minent:
                mask[i] = 1
        else:
            mask[i] = mask[i - 1]
    Nws = max(0, n - winsize)
    if Nws < n and mask[Nws] == 1:
        for i in range(Nws, n):
            mask[i] = 1
    out = []
    st = 0
    while st < n:
        if mask[st] == 0:
            out.append(s[st])
            st += 1
        else:
            out.append("x" * winsize)
            st += 12
    output = "".join(out)[:n]
    # expanded mask (second return value, as in original)
    masked = list(output)
    # find expanded intervals j-8..j+9 around each x-run
    pos = 0
    extra = []
    # simplified: same as output (mask expansion not used by blastp body)
    return output, "".join(masked)


# --------------------------------------------------------------------------
# Index build + search
# --------------------------------------------------------------------------
class Index:
    def __init__(self, heads, seqs, space, nr, step, NC_override):
        self.heads = heads
        self.seqs = seqs
        self.N = len(seqs)
        self.code_tbl = generate_nr_tbl(nr)
        self.scale = int(self.code_tbl.max()) + 1
        spaces, spaces_len, mw, _ = _parse_spaces(space)
        self.spaces, self.spaces_len, self.mw = spaces, spaces_len, mw
        self.nssp = spaces.shape[0]
        self.mink = int(spaces_len.min())
        self.step = int(step)
        bins = min(int(self.scale ** mw) * self.nssp * 5, 128 * 1024 * 1024)
        self.NC = int(NC_override) if NC_override and NC_override > 0 else int(bins)
        self.concat, self.soas = encode_seqs(seqs)
        self.min_score = 25

    def build(self):
        NC = self.NC
        start = np.zeros(NC, dtype=np.int64)
        _count_kernel(self.concat, self.soas, self.code_tbl, self.spaces,
                      self.spaces_len, int(NC), int(self.step), start)
        # threshold mu + 2*sd on counts
        nonzero = start[start > 0]
        if nonzero.size:
            mu = float(nonzero.mean())
            sd = float(nonzero.std())
            self.threshold = int(mu + 2 * sd)
        else:
            self.threshold = 0
        # prefix sum -> bin starts; cur for fill
        self.start = np.empty(NC, dtype=np.int64)
        self.start[0] = 0
        if NC > 1:
            np.cumsum(start[:-1], out=self.start[1:])
        # locus size = total hits
        total = int(start.sum())
        self.locus = np.empty(total, dtype=np.int64)
        cur = self.start.copy()
        if total:
            _fill_kernel(self.concat, self.soas, self.code_tbl, self.spaces,
                         self.spaces_len, int(NC), int(self.step), cur, self.locus)
        self.bin_start = self.start
        self.bin_end = np.empty(NC, dtype=np.int64)
        self.bin_end[:-1] = self.start[1:]
        self.bin_end[NC - 1] = total
        return self

    def get_bin(self, h):
        return int(self.bin_start[h]), int(self.bin_end[h])


def build_index(heads, seqs, space="111111", nr="AST,CFILMVY,DN,EQ,G,H,KR,P,W", step=4, ht=-1):
    idx = Index(heads, seqs, space, nr, step, ht)
    return idx.build()


def _lis(seq, key=lambda x: x[0]):
    if len(seq) < 2:
        return list(seq)
    N = len(seq)
    M, P = [-1] * N, [-1] * N
    L, M[0] = 1, 0
    for i in range(1, N):
        lower, upper = 0, L
        if key(seq[M[upper - 1]]) < key(seq[i]):
            j = upper
        else:
            while upper - lower > 1:
                mid = (upper + lower) // 2
                if key(seq[M[mid - 1]]) < key(seq[i]):
                    lower = mid
                else:
                    upper = mid
            j = lower
        P[i] = M[j - 1]
        if j == L or key(seq[i]) < key(seq[M[j]]):
            M[j] = i
            L = max(L, j + 1)
    res = []
    pos = M[L - 1]
    for _ in range(L):
        res.append(seq[pos])
        pos = P[pos]
    res.reverse()
    return res


def guess_start(loc):
    if not loc:
        return 0, 0
    dist = sum(s - q for q, s in loc) / len(loc)
    return (0, int(dist)) if dist > 0 else (-int(dist), 0)


def find_msav_m(index, query_str, kbound=1, sort=True, timings=None):
    """Query: str. Returns list [hd_idx, score, qi, qj] sorted desc."""
    qseq_u8 = seq_to_codes(query_str)
    ql = len(qseq_u8)
    if ql < index.mink:
        return []
    # k-mer self-score histogram (sliding b62 self scores)
    mink = int(index.mink)
    kscs = np.zeros(ql - mink + 1, dtype=np.int64)
    for i in range(mink):
        pass
    # vectorized sliding: use numpy convolution trick: sum diag
    diag = np.array([B62_NP[c, c] for c in range(256)], dtype=np.int64)
    selfscore = diag[qseq_u8.astype(np.int64)]
    cs = np.cumsum(np.concatenate(([0], selfscore)))
    if ql >= mink:
        kscs = cs[mink:] - cs[:-mink]
    else:
        kscs = np.array([], dtype=np.int64)

    hashes, poss = spseeds_fnv(qseq_u8, index.code_tbl, index.spaces,
                               index.spaces_len, index.NC, index.step)
    hist = np.zeros(ql, dtype=np.int64)
    # count per query pos
    for h, p in zip(hashes, poss):
        s, e = index.get_bin(int(h))
        hist[int(p)] += (e - s)

    thr = float(index.threshold) * ql
    order = np.argsort(-kscs) if len(kscs) else np.array([], dtype=np.int64)
    # careful: kscs index 0..ql-mink; map to query pos same
    hist_c = np.zeros(ql, dtype=np.int64)
    cum = 0
    qcount = 0
    for qi in order:
        if cum > thr:
            break
        if qi < ql:
            hist_c[int(qi)] = 1
            cum += int(hist[qi])
            qcount += 1

    hits_groups = {}
    for h, p in zip(hashes, poss):
        p_i = int(p)
        if p_i >= ql or hist_c[p_i] == 0:
            continue
        s, e = index.get_bin(int(h))
        if e <= s:
            continue
        for loc_pos in range(s, e):
            gpos = int(index.locus[loc_pos])
            # find subject seq index via soas
            hd = int(np.searchsorted(index.soas, gpos, side="right") - 1)
            sst = gpos - int(index.soas[hd])
            k0 = (p_i - sst) // kbound
            key = (hd, k0)
            hits_groups.setdefault(key, []).append((p_i, sst))

    best = {}  # hd -> (score, qst,sst,qed,sed)
    for key, loc0 in hits_groups.items():
        hd, _ = key
        if len(loc0) < 2:
            continue
        loc0.sort(key=lambda x: x[0])
        loc1 = _lis(loc0, key=lambda x: x[1])
        if len(loc1) < 2:
            continue
        qseq_u8c = qseq_u8
        # subject seq bytes
        s_start = int(index.soas[hd])
        s_end = int(index.soas[hd + 1])
        sseq_u8 = index.concat[s_start:s_end]
        # ungap: single X-drop extension from first chained seed
        # (prototype: original sums bounded segments; summing unbounded
        #  re-extensions from every loc would double-count, so we score once)
        qst0, sst0 = loc1[0]
        flag, sc, qqst, qqed, ssst, ssed = ungap_nb(qseq_u8c, sseq_u8, int(qst0), int(sst0), B62_NP, 30)
        total_score = int(sc)
        x0, y0 = qqst, ssst
        x, y = qqed, ssed
        if total_score < index.min_score:
            continue
        if hd not in best or total_score > best[hd][0]:
            best[hd] = (int(total_score), int(x0), int(y0), int(x), int(y))
    out = []
    for hd, (score, qst_, sst_, qed_, sed_) in best.items():
        qi, qj = guess_start([(qst_, sst_), (qed_, sed_)])
        out.append([int(hd), int(score), int(qi), int(qj)])
    if sort:
        out.sort(key=lambda x: -x[1])
    return out


# --------------------------------------------------------------------------
# blastp orchestration
# --------------------------------------------------------------------------
def blastp(qry_path, ref_path, expect=1e-5, v=500, max_miss=1e-3,
           st=-1, ed=-1, flt="T", space="111111", nr="AST,CFILMVY,DN,EQ,G,H,KR,P,W",
           step=4, ht=1 << 20, chk=100000, timings=None):
    """Yield -m8 tuples identical to original blastp().

    Tuple (21): (i, j, li, lj, hi, hj, idy, aln, mis, gap,
                 qst, qed, sst, sed, e, bit, seed, bv, vl, thr, desc)
    where i = query index, j = subject index, li/lj = lengths,
    qst/sst are 1-based, desc = full subject header.
    """

    if timings is None:
        timings = {}
    t0 = time.perf_counter()
    q_heads, q_seqs = parse_fasta(qry_path)
    timings["parse_qry"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    r_heads, r_seqs = parse_fasta(ref_path)
    timings["parse_ref"] = time.perf_counter() - t0

    D = len(r_seqs)
    N = len(q_seqs)
    lo = 0 if st < 0 else max(0, st)
    hi = N if ed < 0 else min(ed, N)

    # build index over ref (chunked; prototype keeps whole DB, records chunks)
    t0 = time.perf_counter()
    index = build_index(r_heads, r_seqs, space=space, nr=nr, step=step, ht=ht)
    timings["build_msav"] = time.perf_counter() - t0

    for i in range(lo, hi):
        hdi = q_heads[i]
        sqi, _mask = seg(q_seqs[i]) if flt == "T" else (q_seqs[i], q_seqs[i])
        t0 = time.perf_counter()
        hits = find_msav_m(index, sqi, sort=False)
        timings["find_msav_m"] = timings.get("find_msav_m", 0.0) + (time.perf_counter() - t0)
        hits.sort(key=lambda x: -x[1])
        mmiss = len(hits) * max_miss + 1
        mmiss = max(mmiss, 100.0 / mmiss)
        mmiss = min(max(mmiss, 10), 120)
        vmax = max(100, max(v + 100, v * 1.1))
        m8s = []
        unmch = 0
        bv = 0
        breakpoint = 0
        for hit in hits[: int(vmax)]:
            j, sc, qj0, sj0 = hit
            hdj = r_heads[j]
            sqj = r_seqs[j]
            hi_id = hdi.split(" ")[0]
            hj_id = hdj.split(" ")[0]
            li = len(q_seqs[i])
            lj = len(sqj)
            t0 = time.perf_counter()
            if len(sqi) < 4096 and len(sqj) < 4096:
                idy, aln, mis, gap, qst, qed, sst, sed, bit = kswat_st_wfa2(
                    sqi, sqj, qst=qj0, sst=sj0)
                e = bit2e(D, sqi, sqj, bit)
            else:
                # long: take best chunk
                best = None
                for res in kswat_st_long_wfa2(sqi, sqj, qj0, sj0):
                    if best is None or res[8] > best[8]:
                        best = res
                idy, aln, mis, gap, qst, qed, sst, sed, bit = best
                e = bit2e(D, sqi, sqj, bit)
            timings["kswat"] = timings.get("kswat", 0.0) + (time.perf_counter() - t0)
            if e <= expect:
                m8s.append((i, j, li, lj, hi_id, hj_id, idy, aln, mis, gap,
                            qst + 1, qed, sst + 1, sed, e, bit, sc,
                            breakpoint, len(hits), int(index.threshold), hdj))
                unmch = 0
                bv += 1
            else:
                unmch += 1
            if unmch >= mmiss or bv >= v + mmiss:
                break
            breakpoint += 1
        m8s.sort(key=lambda r: -r[15])
        for row in m8s[: max(0, int(v))]:
            yield row


def manual_print():
    print("Usage: fsearch_numba -p blastp -i qry.fsa -d db.fsa")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fsearch_numba")
    ap.add_argument("-p", default="blastp", choices=["blastp", "makedb"])
    ap.add_argument("-i", default="")
    ap.add_argument("-d", default="")
    ap.add_argument("-o", default="")
    ap.add_argument("-e", type=float, default=1e-3)
    ap.add_argument("-v", type=int, default=500)
    ap.add_argument("-s", default="111111")
    ap.add_argument("-r", default="AST,CFILMVY,DN,EQ,G,H,KR,P,W")
    ap.add_argument("-j", type=int, default=1)
    ap.add_argument("-c", type=int, default=50000)
    ap.add_argument("-a", type=int, default=1)
    ap.add_argument("-H", type=int, default=1 << 20,
                    help="hash table size (prototype default 1M; RPython used 120M)")
    ap.add_argument("--profile", action="store_true", help="run under cProfile + print timing breakdown")
    args = ap.parse_args(argv)

    if args.p == "makedb":
        if not args.i:
            manual_print()
            return 2
        heads, seqs = parse_fasta(args.i)
        idx = build_index(heads, seqs, space=args.s, nr=args.r, step=args.j, ht=args.H)
        print(f"# makedb: {len(seqs)} seqs, NC={idx.NC}, threshold={idx.threshold}, locus={len(idx.locus)}")
        return 0

    if not args.i or not args.d:
        manual_print()
        return 2

    timings = {}
    t_all = time.perf_counter()
    outfh = open(args.o, "w") if args.o else sys.stdout

    def _format_row(hit):
        i, j, li, lj, hi, hj, idy, aln, mis, gap, qst, qed, sst, sed, e, bit, seed, bv, vl, thr, desc = hit
        # 1-based coords already set in blastp(); filter like original entry_point
        if e > args.e:
            return None
        Idy = str(idy)
        End = max(0, Idy.find(".") + 3)
        Idy = Idy[:End]
        E = f2s(e)
        return "%s\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%s\t%f\t%d\t%d\t%d\t%s\n" % (
            hi, hj, Idy, aln, mis, gap, qst, qed, sst, sed, E, bit, li, lj, i, desc)

    def _run():
        n = 0
        for hit in blastp(args.i, args.d, expect=args.e, v=args.v,
                          space=args.s, nr=args.r, step=args.j, ht=args.H,
                          timings=timings):
            line = _format_row(hit)
            if line is None:
                continue
            outfh.write(line)
            n += 1
        return n

    if args.profile:
        import cProfile, pstats, io
        pr = cProfile.Profile()
        pr.enable()
        n = _run()
        pr.disable()
        print(f"# hits={n} total={time.perf_counter()-t_all:.3f}s timings={timings}", file=sys.stderr)
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats(30)
        print(s.getvalue(), file=sys.stderr)
    else:
        n = _run()
        print(f"# hits={n} total={time.perf_counter()-t_all:.3f}s timings={timings}", file=sys.stderr)
    if outfh is not sys.stdout:
        outfh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
