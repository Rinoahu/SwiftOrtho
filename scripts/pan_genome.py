#!usr/bin/env python
import sys
import os
import itertools
from itertools import izip
#import matplotlib.pyplot as plt
from Bio import SeqIO
import scipy as np
from scipy import median, mean
from scipy.optimize import curve_fit
from scipy.stats.distributions import t
from struct import pack
from random import shuffle, seed
from itertools import combinations
try:
    from sympy.mpmath import fac
except:
    from math import factorial as fac
from random import sample
try:
    from numba import jit
except:
    jit = lambda x:x

try:
    from numexpr import evaluate
except:
    evaluate = eval


# do the core gene find
# python this_script.py -i foo.pep.fsa -c foo.mcl [-l .5] [-u .95]
def manual_print():
    print 'Usage:'
    print '  python this_script.py -i foo.pep.fsa -g foo.mcl [-l .05] [-u .95]'
    print 'Parameters:'
    print ' -i: protein/gene fasta file. The header should be like xxxx|yyyy: xxxx is taxon name and yyyy is unqiue identifier'
    print ' -g: protein/gene groups file. The proteins of each raw belong to the same protein/gene group'
    print ' -l: threshold for specific genes. Default 0.05'
    print ' -u: threshold for core genes. Default[0.95]'
    print ' -r: a file contains the species names that are used for pan-genome analysis.'


argv = sys.argv
# recommand parameter:
args = {'-i':'', '-g':'', '-l':.05, '-u':.95, '-r':None}

N = len(argv)
for i in xrange(1, N):
    k = argv[i]
    if k in args:
        v = argv[i+1]
        args[k] = v
    elif k[:2] in args and len(k) > 2:
        args[k[:2]] = k[2:]
    else:
        continue

if args['-i']=='' or args['-g']=='':
    manual_print()
    raise SystemExit()

try:
    fas, mcl, ts, tc, ftax = args['-i'], args['-g'], float(args['-l']), float(args['-u']), args['-r']
except:
    manual_print()
    raise SystemExit()


tax_allow = set()
if ftax and os.path.isfile(ftax):
    f = open(ftax, 'r')
    for i in f:
        tax_allow.add(i.strip())
    f.close()


# get all the taxon
taxon_set = set()
f = open(fas, 'r')
for i in f:
    if i.startswith('>'):
        header = i[1:-1].split(' ')[0]
        tax = header.split('|')[0]
        if tax not in tax_allow and tax_allow:
            continue
        taxon_set.add(tax)

f.close()

taxon_list = list(taxon_set)
taxon_dict = {j:i for i,j in enumerate(taxon_list)}



# build N by M matrix
# row is group name
# col is taxon name
# each cell stands for the genes count of the group for a specific taxon
#header = ['#family', 'type'] + taxon_list
_o0 = open('type.txt', 'wb')
_o1 = open('pan.npy', 'wb')
#print '\t'.join(header)
N = len(taxon_list)
Ts = ts < 1 and max(ts*N, 1) or ts
Tc = tc < 1 and tc * N or tc

#outputs = []
spec = shar = core = 0
visit = set()
flag = 0
f = open(mcl, 'r')
for i in f:
    counts = [0] * N
    #taxon_dict[elem.split('|')[0]] for elem in i[:-1].split('\t')]
    group = i[:-1].split('\t')
    for j in group:
        tax = j.split('|')[0]
        if tax not in tax_allow and tax_allow:
            continue
        counts[taxon_dict[tax]]+=1
        visit.add(j)

    #thr = len([elem for elem in counts if elem>0]) * 1. / N
    thr = len([elem for elem in counts if elem>0]) 
    if thr <= Ts:
        pan = 'Specific'
        spec += 1
    elif Ts< thr < Tc:
        pan = 'Share'
        shar += 1
    else:
        pan = 'Core'
        core += 1

    #output = [flag, pan]
    output = ['group_%09d'%flag, pan]
    output.extend(counts)

    _o0.write('\t'.join(output[:2])+'\n')
    _o1.write(''.join([pack('i', elem) for elem in counts]))
    #print '\t'.join(map(str, output))
    #outputs.append(output)
    flag += 1

f.close()

for i in SeqIO.parse(fas, 'fasta'):
    j = i.id
    if j in visit:
        continue
    counts = [0] * N
    tax = j.split('|')[0]

    if tax not in tax_allow and tax_allow:
        continue

    counts[taxon_dict[tax]]+=1
    pan = 'Specific'
    output = ['group_%09d'%flag, pan]
    output.extend(counts)
    _o0.write('\t'.join(output[:2])+'\n')
    _o1.write(''.join([pack('i', elem) for elem in counts]))
    #print '\t'.join(map(str, output))
    #outputs.append(output)
    flag += 1
    spec += 1


_o0.close()
_o1.close()

print '#' * 80
print '# Statistics and profile of pan-genome:'
print '# The methods can be found in Hu X, et al. Trajectory and genomic determinants of fungal-pathogen speciation and host adaptation.'

print '#'
print '# statistic of core, shared and specific genes:'
print '\t'.join(['# Feature', 'core', 'shared', 'specific', 'taxon'])
print '\t'.join(map(str, ['# Number', core, shar, spec, N]))
#print flag, N

#_o.close()
# calculate the core, share and specific gene's profile
fp = np.memmap('pan.npy', mode='r+', shape=(flag, N), dtype='int32')
mat = np.asarray(fp, dtype='bool')
mat = np.asarray(mat, dtype='int8')
#mat[mat>0] = 1
#mat = np.asarray([elem[2:] for elem in outputs])

#print ts, tc, Ts, Tc
#print mat
def pan_feature0(x, ts=.05, tc=.95):
    n, d = x.shape
    idx = range(d)
    index = []
    cores = []
    specs = []
    panzs = []
    for i in xrange(1, d+1):
        j = i+1
        Ts = ts < 1 and max(ts*j, 1) or ts
        Tc = tc < 1 and tc * j or tc
        #shuffle(idx)
        flag = 0
        for j in combinations(idx, i):
            if flag > 100:
                break
            y = x[:,j].sum(1)
            core = np.sum(y>Tc)
            spec = np.sum(y<Ts)
            panz = np.sum(y>0)
            index.append(i)
            cores.append(core)
            specs.append(spec)
            panzs.append(panz)
            flag += 1

    return index, cores, specs, panzs

def pan_feature1(x, size=100, ts=.05, tc=.95):
    n, d = x.shape 
    idx = range(d)
    index = []
    cores = []
    specs = []
    panzs = []
    for itr in xrange(size):
        shuffle(idx)
        y = mat[:, idx[0]]
        for i in xrange(1, d-1):
            j = i+1
            Ts = ts < 1 and max(ts*j, 1) or ts
            Tc = tc < 1 and tc * j or tc
            y += mat[:, idx[i]]
            core = np.sum(y>=Tc)
            spec = np.sum(np.logical_and(y<=Ts, y>0))
            panz = np.sum(y>0)
            index.append(j)
            cores.append(core)
            specs.append(spec)
            panzs.append(panz)

    return index, cores, specs, panzs



def pan_feature(x, size=100, ts=.05, tc=.95):
    n, d = x.shape 
    #size = min(size, d*(d-1)/2)
    idx = range(d)
    index = []
    cores = []
    specs = []
    panzs = []
    idxs = []
    seed(42)
    for i in xrange(size):
        shuffle(idx)
        idxs.append(idx[:])

    ys = x[:, [elem[0] for elem in idxs]]
    #for i in xrange(1, d-1):
    for i in xrange(1, d):
        j = i + 1
        Ts = ts < 1 and max(ts*j, 1) or ts
        Tc = tc < 1 and tc * j or tc

        yn = x[:, [elem[i] for elem in idxs]]
        sp = np.asarray(evaluate('(ys<=0) & (yn>0)'), dtype='int8')
        #sp = np.asarray(evaluate('(ys<=Ts) & (yn>0)'), dtype='int8')
        spec = evaluate('sum(sp, 0)')

        ys = evaluate('ys+yn')
        cr = np.asarray(evaluate('ys>=Tc'), dtype='int8')
        core = evaluate('sum(cr, 0)')
        #core = evaluate('sum(Ys>=Tc, 0)')
        #core = np.sum(ys>=Tc, 0)
        #spec = evaluate('sum((Ys<=Ts) & (Ys>0), 0)')
        #spec = np.sum((ys<=Ts) & (ys>0), 0)
        pa = np.asarray(evaluate('ys>0'), dtype='int8')
        panz = evaluate('sum(pa, 0)')
        #panz = np.sum(ys>0, 0)

        cores.extend(core)
        specs.extend(spec)
        panzs.extend(panz)
        index.extend([j] * size)

        '''
        if i < d-1:
            cores.extend(core)
            specs.extend(spec)
            panzs.extend(panz)
            index.extend([j] * size)
        else:
            cores.extend(core[:1])
            specs.extend(spec[:1])
            panzs.extend(panz[:1])
            index.extend([j])
        '''

        #print 'ts tc ys'
        #print Ts, Tc
        #print ys
        #print [elem[i] for elem in idxs]
        #print 'core spec panz', len(core), len(spec), len(panz)
        #print core, spec, panz
        #print [j] * size

    #print 'pan genome'
    #print map(len, [index, cores, specs, panzs])
    #print index
    #print cores
    #print specs
    #print panzs
    return index, cores, specs, panzs



index, cores, specs, panzs = pan_feature(mat)

#for a, b in zip(index, specs):
#    print a, b

#raise SystemExit()

# compute the combine
def combs(N, M):
    return fac(N) / fac(M) / fac(N - M)

# plt.figure()
#plt.plot(x, y, label = '$sin(x)$', color = 'red', linewidth = 1)
# plt.show()

# estimate core gene size
def Fc(n, K_c, Tau_c, Omega):
    #return K_c * np.exp(-n / Tau_c) + Omega
    return K_c * np.exp(-n / max(1e-30, Tau_c)) + Omega

# estimate specific gene size
def Fs(n, K_s, Tau_s, TgTheta):
    #return K_s * np.exp(-n / Tau_s) + TgTheta
    return K_s * np.exp(-n / max(1e-30, Tau_s)) + min(1e8, TgTheta)

# pan-genome open test
# alpah <= 1 is open
# alpha > 1 is close
#def pan_open(n, K, Alpha):
#    return K * n ** (-Alpha)

# pan size
def fpan(n, D, tgTheta, K_s, Tau_s):
    return D + tgTheta * (n - 1) + K_s * np.exp(-2. / Tau_s) * (1 - np.exp(-(n - 1.) / Tau_s)) / (1 - np.exp(-1. / Tau_s))

# estimate pan-genome gene size
# r > 0 : open
def pgene(n, K, r):
    return K * n ** r

def find_med(coreN):
    med = {}
    for i, j in coreN:
        try:
            med[i].append(j)
        except:
            med[i] = [j]
    for i in med:
        med[i] = np.median(med[i])
    return np.asarray(med.items(), 'int64')


def fit_curve(f, X, Y, alpha=.05):
    x, y = map(np.asarray, [X, Y])
    pars, pcov = curve_fit(f, x, y)
    n = len(y)
    p = len(pars)
    dof = max(0, n - p)
    tval = t.ppf(1.0 - alpha / 2., dof)
    conf = [tval * elem ** .5 for elem in np.diag(pcov)]
    return pars, conf

#pm = '+/-'
pm = '\xc2\xb1'
#spcN = [elem for elem in coreN if elem[0] == 1] + spcN

# estimate the parameters
# the core parameter
# print 'the core N', coreN
#coreN = find_med(coreN)
# print 'the core N', coreN.tolist()
#popt, pcov = curve_fit(Fc, coreN[:, 0], coreN[:, 1])
#popt, conf = fit_curve(Fc, num, coreN[:, 1])\
print '#'
print '# \xcf\x89(core size of pan-genome) and 95% confidence interval:'
popt, conf = fit_curve(Fc, index, cores)
#print 'Kc\tTauc\tOmega', popt, conf
print '# \xce\xbac\t\xcf\x84c\t\xcf\x89'
#print pm
print '# '+'\t'.join([str(a)+pm+str(b) for a, b in zip(popt, conf)])


# the specific parameter
# print 'the spc N', spcN
#spcN = find_med(spcN)
#spcN = np.asarray(spcN, 'int64')
# print 'the spc N', spcN.tolist()
#popt, pcov = curve_fit(Fs, spcN[:, 0], spcN[:, 1])
#popt, conf = fit_curve(Fs, spcN[:, 0], spcN[:, 1])

print '#'
print '# \xce\xb8(new gene number for everay new genome sequenced) and 95% confidence interval:'
popt, conf = fit_curve(Fs, index, specs)
#print '# Ks\tTaus\tTheta', popt, conf
print '# \xce\xbas\t\xcf\x84s\ttg(\xce\xb8)'
print '# '+'\t'.join([str(a)+pm+str(b) for a, b in zip(popt, conf)])



# the openness
#print '#'
#print '# \xce\xb1(parameter of openness test) and 95% confidence interval(open if \xce\xb1 <= 1 else close):'
#popt, conf = fit_curve(pan_open, index, specs)
#print '# K\tAlpah', popt, conf
#print '# \xce\xba\t\xce\xb1 '
#print '# '+'\t'.join([str(a)+pm+str(b) for a, b in zip(popt, conf)])



# the pan-genome size
#pan_size = np.asarray(pan_size, 'int64')
#popt, pcov = curve_fit(pgene, pan_size[:, 0], pan_size[:, 1])
#popt, conf = fit_curve(pgene, pan_size[:, 0], pan_size[:, 1])

print '#'
print '# \xce\xba(size and openess of pan-genome, open if \xce\xb3 > 0) and 95% confidence interval:'
popt, conf = fit_curve(pgene, index, panzs)
#print 'pan-size, k, gamma', popt, conf
print '# \xce\xba\t\xce\xb3'
print '# '+'\t'.join([str(a)+pm+str(b) for a, b in zip(popt, conf)])

print '#'
print '# Type and frequency of each gene group in different species:'
print '#'*80
header = ['#family', 'type'] + taxon_list
#_o = open('pan.npy', 'wb')
print '\t'.join(header)
#for i in outputs:
#    print '\t'.join(map(str, i))
f = open('type.txt', 'r')
#mat = np.memmap('pan.npy', mode='r+', shape=(flag, N), dtype='int32')
for i, j in izip(f, fp):
    out = i[:-1] + '\t' + '\t'.join(map(str, j))
    print out

f.close()
os.system('rm pan.npy type.txt')
#print fp
