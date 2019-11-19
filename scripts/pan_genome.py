#!usr/bin/env python
import sys
import os
import itertools

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
    jit = lambda x: x

try:
    from numexpr import evaluate
    cpu = 1
except:
    evaluate = eval
    cpu = 0

# do the core gene find
# python this_script.py -i foo.pep.fsa -c foo.mcl [-l .5] [-u .95]


def manual_print():
    print('Usage:')
    print(
        '  python this_script.py -i foo.pep.fsa -g foo.mcl [-l .05] [-u .95]')
    print('Parameters:')
    print(' -i: protein/gene fasta file. The header should be like xxxx|yyyy: xxxx is taxon name and yyyy is unqiue identifier')
    print(' -g: protein/gene groups file. The proteins of each raw belong to the same protein/gene group')
    print(' -l: threshold for specific genes. Default 0.05')
    print(' -u: threshold for core genes. Default[0.95]')
    print(' -r: a file contains the species names that are used for pan-genome analysis.')


argv = sys.argv
# recommand parameter:
args = {'-i': '', '-g': '', '-l': .05, '-u': .95, '-r': None}

N = len(argv)
for i in range(1, N):
    k = argv[i]
    if k in args:
        v = argv[i + 1]
        args[k] = v
    elif k[:2] in args and len(k) > 2:
        args[k[:2]] = k[2:]
    else:
        continue

if args['-i'] == '' or args['-g'] == '':
    manual_print()
    raise SystemExit()

try:
    fas, mcl, ts, tc, ftax = args[
        '-i'], args['-g'], float(args['-l']), float(args['-u']), args['-r']
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
taxon_dict = {j: i for i, j in enumerate(taxon_list)}


# build N by M matrix
# row is group name
# col is taxon name
# each cell stands for the genes count of the group for a specific taxon
# header = ['#family', 'type'] + taxon_list
_o0 = open('type.txt', 'wb')
_o1 = open('pan.npy', 'wb')
# print '\t'.join(header)
N = len(taxon_list)
Ts = ts < 1 and max(ts * N, 1) or ts
Tc = tc < 1 and tc * N or tc

#outputs = []
spec = shar = core = 0
visit = set()
flag = 0
f = open(mcl, 'r')
for i in f:
    counts = [0] * N
    # taxon_dict[elem.split('|')[0]] for elem in i[:-1].split('\t')]
    group = i[:-1].split('\t')
    for j in group:
        tax = j.split('|')[0]
        if tax not in tax_allow and tax_allow:
            continue
        counts[taxon_dict[tax]] += 1
        visit.add(j)

    #thr = len([elem for elem in counts if elem>0]) * 1. / N
    thr = len([elem for elem in counts if elem > 0])
    if thr <= Ts:
        pan = 'Specific'
        spec += 1
    elif Ts < thr < Tc:
        pan = 'Share'
        shar += 1
    else:
        pan = 'Core'
        core += 1

    #output = [flag, pan]
    output = ['group_%09d' % flag, pan]
    output.extend(counts)

    output_2 = '\t'.join(output[:2]) + '\n'
    _o0.write(output_2.encode())

    #_o0.write('\t'.join(output[:2])+'\n')

    #output_3 = ''.join([pack('i', elem).decode() for elem in counts])
    output_3 = b''.join([pack('i', elem) for elem in counts])
    #_o1.write(output_3.encode())
    _o1.write(output_3)

    #_o1.write(''.join([pack('i', elem) for elem in counts]))
    # print '\t'.join(map(str, output))
    # outputs.append(output)
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

    counts[taxon_dict[tax]] += 1
    pan = 'Specific'
    output = ['group_%09d' % flag, pan]
    output.extend(counts)

    output_0 = '\t'.join(output[:2]) + '\n'
    _o0.write(output_0.encode())

    #_o0.write('\t'.join(output[:2])+'\n')

    output_1 = ''.join([pack('i', elem).decode() for elem in counts])
    _o1.write(output_1.encode())
    #_o1.write(''.join([pack('i', elem) for elem in counts]))

    # print '\t'.join(map(str, output))
    # outputs.append(output)
    flag += 1
    spec += 1


_o0.close()
_o1.close()

print('#' * 80)
print('# Statistics and profile of pan-genome:')
print('# The methods can be found in Hu X, et al. Trajectory and genomic determinants of fungal-pathogen speciation and host adaptation.')

print('#')
print('# statistic of core, shared and specific genes:')
print('\t'.join(['# Feature', 'core', 'shared', 'specific', 'taxon']))
print('\t'.join(map(str, ['# Number', core, shar, spec, N])))
# print flag, N

#_o.close()
# calculate the core, share and specific gene's profile
#print('flag and N', flag, N)
fp = np.memmap('pan.npy', mode='r+', shape=(flag, N), dtype='int32')

# print 'fp is', fp[:]
mat = np.asarray(fp, dtype='bool')
mat = np.asarray(mat, dtype='int8')
#mat[mat>0] = 1
#mat = np.asarray([elem[2:] for elem in outputs])

# print ts, tc, Ts, Tc
# print mat


def pan_feature0(x, ts=.05, tc=.95):
    n, d = x.shape
    idx = list(range(d))
    index = []
    cores = []
    specs = []
    panzs = []
    for i in range(1, d + 1):
        j = i + 1
        Ts = ts < 1 and max(ts * j, 1) or ts
        Tc = tc < 1 and tc * j or tc
        # shuffle(idx)
        flag = 0
        for j in combinations(idx, i):
            if flag > 100:
                break
            y = x[:, j].sum(1)
            core = np.sum(y > Tc)
            spec = np.sum(y < Ts)
            panz = np.sum(y > 0)
            index.append(i)
            cores.append(core)
            specs.append(spec)
            panzs.append(panz)
            flag += 1

    return index, cores, specs, panzs


def pan_feature1(x, size=20, ts=.05, tc=.95):
    n, d = x.shape
    idx = list(range(d))
    index = []
    cores = []
    specs = []
    panzs = []
    for itr in range(size):
        shuffle(idx)
        y = mat[:, idx[0]]
        for i in range(1, d - 1):
            j = i + 1
            Ts = ts < 1 and max(ts * j, 1) or ts
            Tc = tc < 1 and tc * j or tc
            y += mat[:, idx[i]]
            core = np.sum(y >= Tc)
            spec = np.sum(np.logical_and(y <= Ts, y > 0))
            panz = np.sum(y > 0)
            index.append(j)
            cores.append(core)
            specs.append(spec)
            panzs.append(panz)

    return index, cores, specs, panzs


def pan_feature(x, size=100, ts=.05, tc=.95):
    # print 'x shape', x.shape, x.min()
    n, d = x.shape
    #size = min(size, d*(d-1)/2)
    idx = list(range(d))
    index = []
    cores = []
    specs = []
    panzs = []
    idxs = []
    seed(42)
    for i in range(size):
        shuffle(idx)
        idxs.append(idx[:])

    #print('idxs', idxs)
    ys = np.asarray(x[:, [elem[0] for elem in idxs]] > 0, 'int32')
    #ys_0 = np.asarray(x[:, idxs[0][0]] > 0, 'int32')

    # for i in xrange(1, d-1):
    for i in range(1, d):
        j = i + 1
        Ts = ts < 1 and max(ts * j, 1) or ts
        Tc = tc < 1 and tc * j or tc
        #Tc = max(1, Tc)
        #Ts = 1
        #Tc = j

        #print('iter', j, Ts, ts, Tc, tc)
        #yn_0 = np.asarray(x[:, idxs[0][i]] > 0, 'int32')
        #print('iter', j, ys_0.shape, Ts, Tc, 'core', ((ys_0+yn_0)>=Tc).sum(), 'specific', ((ys_0==0) & (yn_0>0)).sum(), 'before', (ys_0>0).sum(), 'after', ((ys_0+yn_0)>0).sum())
        #ys_0 += yn_0

        yn = np.asarray(x[:, [elem[i] for elem in idxs]] > 0, 'int32')
        if cpu == 1:
            #ys = evaluate('ys + yn')
            #sp = np.asarray(evaluate('(ys == 0) & (yn > 0)'), dtype='int8')
            sp = np.asarray(evaluate('(ys <= Ts) & (yn > 0)'), dtype='int8')
            spec = evaluate('sum(sp, 0)')

            ys = evaluate('ys + yn')

            cr = np.asarray(evaluate('ys >= Tc'), dtype='int8')
            core = evaluate('sum(cr, 0)')
            #core = evaluate('sum(Ys>=Tc, 0)')
            #core = np.sum(ys>=Tc, 0)
            #spec = evaluate('sum((Ys<=Ts) & (Ys>0), 0)')
            #spec = np.sum((ys<=Ts) & (ys>0), 0)
            pa = np.asarray(evaluate('ys > 0'), dtype='int8')
            panz = evaluate('sum(pa, 0)')
            #panz = np.sum(ys>0, 0)
        else:
            #ys = ys + yn
            sp = np.asarray((ys <= Ts-1) & (yn > 0), 'int8')
            #sp = np.asarray((ys == 0) & (yn > 0), 'int8')
            spec = sp.sum(0)

            ys = ys + yn
            cr = np.asarray(ys >= Tc, dtype='int8')
            core = cr.sum(0)
            pa = np.asarray(ys > 0, dtype='int8')
            panz = pa.sum(0)

        #print('means', j, np.mean(core), np.mean(spec), np.mean(panz))
        #core=sorted(core)[::-1]
        #spec=sorted(spec)[::-1]
        #panz=sorted(panz)[::-1]
        end = size

        cores.extend(core[:end])
        specs.extend(spec[:end])
        panzs.extend(panz[:end])
        index.extend(([j] * size)[:end])

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

        # print 'ts tc ys'
        # print Ts, Tc
        # print ys
        # print [elem[i] for elem in idxs]
        # print 'core spec panz', len(core), len(spec), len(panz)
        # print core, spec, panz
        #print [j] * size

    # print 'pan genome'
    # print map(len, [index, cores, specs, panzs])
    # print index
    # print cores
    # print specs
    # print panzs
    return index, cores, specs, panzs


index, cores, specs, panzs = pan_feature(mat, 20, ts, tc)

# print 'index', index
# print 'cores', cores
# print 'specs', specs

_o = open(mcl+'_xy.txt', 'w')
for a, b, c, d in zip(index, cores, specs, panzs):
    #print('test', a, b, c, d)
    abcd = '\t'.join(map(str, [a, b, c, d]))
    _o.write('%s\n'%abcd)

_o.close()

#raise SystemExit()

# compute the combine


def combs(N, M):
    return fac(N) / fac(M) / fac(N - M)

# plt.figure()
#plt.plot(x, y, label = '$sin(x)$', color = 'red', linewidth = 1)
# plt.show()

# estimate core gene size


def Fc(n, K_c, Tau_c, Omega):
    return K_c * np.exp(-n / Tau_c) + Omega
    #return K_c * np.exp(-n / max(1e-30, Tau_c)) + Omega

# estimate specific gene size


def Fs(n, K_s, Tau_s, TgTheta):
    return K_s * np.exp(-n / Tau_s) + TgTheta
    #return K_s * np.exp(-n / max(1e-30, Tau_s)) + min(1e8, TgTheta)

# pan-genome open test
# alpah <= 1 is open
# alpha > 1 is close
# def pan_open(n, K, Alpha):
#    return K * n ** (-Alpha)

# pan size

# estimate how many new genes  will be found after a new sequenced genome
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
    return np.asarray(list(med.items()), 'int64')

def fit_curve(f, X, Y, alpha=.05, bounds=None):
    x, y = list(map(np.asarray, [X, Y]))
    if bounds:
        try:
            # print x, y
            pars, pcov = curve_fit(f, x, y, bounds=bounds)
        except:
            pars, pcov = curve_fit(f, x, y, method='dogbox', bounds=bounds)

    else:
        try:
            # print x, y
            pars, pcov = curve_fit(f, x, y)
        except:
            pars, pcov = curve_fit(f, x, y, method='dogbox')

    n = len(y)
    p = len(pars)
    dof = max(0, n - p)
    tval = t.ppf(1.0 - alpha / 2., dof)
    conf = [tval * elem ** .5 for elem in np.diag(pcov)]
    return pars, conf

params = []
#pm = '+/-'
#pm = '\xc2\xb1'
#pm = '±'
pm = chr(177)
#spcN = [elem for elem in coreN if elem[0] == 1] + spcN

# estimate the parameters
# the core parameter
# print 'the core N', coreN
#coreN = find_med(coreN)
# print 'the core N', coreN.tolist()
#popt, pcov = curve_fit(Fc, coreN[:, 0], coreN[:, 1])
#popt, conf = fit_curve(Fc, num, coreN[:, 1])\

# special symbol
k_ = chr(954)
t_ = chr(964)
w_ = chr(969)


print('#')
#print('# ω (core size of pan-genome) and 95% confidence interval:')
print('# ' + w_ + '(core size of pan-genome) and 95% confidence interval:')
popt, conf = fit_curve(Fc, index, cores, bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
# print 'Kc\tTauc\tOmega', popt, conf
#print('# \xce\xbac\t\xcf\x84c\t\xcf\x89')
#print('#\tκc\tτc\tω')
print('#\t%sc\t%sc\t%s'%(k_, t_, w_))

# print pm
print('# ' + '\t'.join([str(a) + pm + str(b) for a, b in zip(popt, conf)]))

params.extend(popt)

# the specific parameter
# print 'the spc N', spcN
#spcN = find_med(spcN)
#spcN = np.asarray(spcN, 'int64')
# print 'the spc N', spcN.tolist()
#popt, pcov = curve_fit(Fs, spcN[:, 0], spcN[:, 1])
#popt, conf = fit_curve(Fs, spcN[:, 0], spcN[:, 1])
theta = chr(952)

print('#')
#print('# θ (new gene number for every new genome sequenced) and 95% confidence interval:')
print('# ' + theta + '(new gene number for each new sequenced genome) and 95% confidence interval:')
popt, conf = fit_curve(Fs, index, specs, bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
# print '# Ks\tTaus\tTheta', popt, conf
#print('# κs\tτs\ttg(θ)')
print('# %ss\t%ss\ttg(%s)'%(k_, t_, theta))

print('# ' + '\t'.join([str(a) + pm + str(b) for a, b in zip(popt, conf)]))
params.extend(popt)


# the openness
# print '#'
# print '# \xce\xb1(parameter of openness test) and 95% confidence interval(open if \xce\xb1 <= 1 else close):'
#popt, conf = fit_curve(pan_open, index, specs)
# print '# K\tAlpah', popt, conf
# print '# \xce\xba\t\xce\xb1 '
# print '# '+'\t'.join([str(a)+pm+str(b) for a, b in zip(popt, conf)])


# the pan-genome size
#pan_size = np.asarray(pan_size, 'int64')
#popt, pcov = curve_fit(pgene, pan_size[:, 0], pan_size[:, 1])
#popt, conf = fit_curve(pgene, pan_size[:, 0], pan_size[:, 1])

print('#')
#print('# κ (size and openess of pan-genome, open if γ > 0) and 95% confidence interval:')
print('# ' + k_ + '(size and openess of pan-genome, open if γ > 0) and 95% confidence interval:')

popt, conf = fit_curve(pgene, index, panzs)
# print 'pan-size, k, gamma', popt, conf
r_ = chr(947)
#print('# κ\tγ')
print('# %s\t%s'%(k_, r_))

print('# ' + '\t'.join([str(a) + pm + str(b) for a, b in zip(popt, conf)]))

params.extend(popt)


print('#')
print('# Type and frequency of each gene group in different species:')
print('#' * 80)
header = ['#family', 'type'] + taxon_list
#_o = open('pan.npy', 'wb')
print('\t'.join(header))
# for i in outputs:
#    print '\t'.join(map(str, i))
f = open('type.txt', 'r')
#mat = np.memmap('pan.npy', mode='r+', shape=(flag, N), dtype='int32')
for i, j in zip(f, fp):
    out = i[:-1] + '\t' + '\t'.join(map(str, j))
    print(out)

f.close()
os.system('rm pan.npy type.txt')

# print fp
#print(params)

curdir = os.getcwd()

# plot the pan-genome feature
Rcmd = '''#!usr/bin/env Rscript
dat<-read.delim('{fname}', sep='\\t', header=F)

end = {end}
y = dat$V2
x = dat$V1

# core genes
a = {a0_}
b = {b0_}
c = {c0_}
fc <- function(n)(a * exp(-n/b) + c)

pdf("{path}/pan_curve.pdf")
par(mfrow=c(2,2))

plot(x,y, xlab='# of genomes', ylab='# of core genes', pch=19)
lines(fc(1:end), col='red', lwd=3)

# new genes per sequenced
K_s = {a1_}
Tau_s = {b1_}
TgTheta = {c1_}
fs <- function(n)(K_s * exp(-n / Tau_s) + TgTheta)

y=dat$V3
x=dat$V1

plot(x,y, xlab='# of genomes', ylab='# of new genes', pch=19)
lines(fs(1:end), col='blue', lwd=3)

# pangenome size
K={a2_}
r={b2_}

fp <- function(n)(K * n ** r)

y=dat$V4
x=dat$V1

plot(x,y, xlab='# of genomes', ylab='size of pan-genome', pch=19)
lines(fp(1:end), col='green', lwd=3)

dev.off()

'''.format(fname=curdir + '/' + mcl+'_xy.txt', a0_=params[0], b0_=params[1], c0_=params[2], a1_=params[3], b1_=params[4], c1_=params[5], a2_=params[6], b2_=params[7], end=max(index), path=curdir)

#print(curdir + '/' + mcl+'_xy.txt')
#print(Rcmd)

if os.system('which Rscript') == 0:
    #print('hello')
    _o = open('./plot_pan.rs', 'w')
    _o.write(Rcmd)
    _o.close()
    os.system('Rscript ./plot_pan.rs')
    #os.system("R -e \'%s\'"%Rcmd)
    os.system('rm ./plot_pan.rs')
    os.system('rm {fname}'.format(fname=curdir + '/' + mcl+'_xy.txt'))


