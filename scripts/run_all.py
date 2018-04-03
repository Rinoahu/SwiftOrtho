#!usr/bin/env python
import sys
#from Bio import SeqIO
from collections import Counter
import os
from time import time

# do the core gene find
# python this_script.py -i foo.pep.fsa -g foo.ortholog [-r taxon]
def manual_print():
    print 'Integrate pipeline can be used to identify orthologs, construct phylotree and get profile of pan-genomes'
    print 'If the operonic information supplied, this pipeline can also cluster the operon_cluster'
    print 'Usage:'
    #print '  python this.py -i foo.pep.fsa [-r taxonomy] [-p foo.operon] [...]'
    print '  python %s -i foo.pep.fsa [-r taxonomy] [-p foo.operon] [...]'%sys.argv[0]


    print 'Input fasta file:'
    print ' -i: protein sequences in fasta formt. The header should be like xxxx|yyyy: xxxx is taxon name and yyyy is unqiue identifier in that taxon'
    print ''

    print 'Operon information if operon clustering is needed:'
    print ' -p: operonic annotation file. The 1st column of this file should be like x0-->x1-->x2-->x3 or x0<--x1<--x2<--x3.'
    print ''


    print 'Optional parameters for all-vs-all homologous search:'
    print ' -s: spaced seed in format: 1111,11110,1001111 parameter'
    print ' -a: number of processors to use'
    print ''

    print 'Optional parameters for orthology inference:'
    print ' -c: min coverage of sequence [0~1]'
    print ' -y: identity [0~100]'
    print ' -n: normalization score [no|bsr|bal]. bsr: bit sore ratio; bal:  bit score over anchored length. Default: no'
    print ''

    print 'Optional parameters for clustering:'
    print ' -A: clustering algorithm. [mcl|apc]'
    print ''

    print 'Optional parameters for species tree construction:'
    print ' -r: taxonomy name used as reference [optional]'
    print ''

    print 'Optional parameters for pan-genome:'
    print ' -l: threshold to identify specific genes. For example, 0.05 means the specific gene should only exist in less than 5% of sll selected pecies'
    print ' -u: threshold to identify core genes. For example, 0.95 means core gene should exist in atleast 95% of all selected species '
    print ' -I: Inflation parameter of mcl. default: 1.5'



argv = sys.argv
# recommand parameter:
args = {'-i': '', '-r': '', '-p': '', '-s':'111111', '-c':'.5', '-y':'50', '-n':'no', '-l':'.05', '-u':'.95', '-a':'4', '-A':'apc', '-I':'1.5'}

N = len(argv)
for i in xrange(1, N):
    k = argv[i]
    if k in args:
        v = argv[i + 1]
        args[k] = v
    elif k[:2] in args and len(k) > 2:
        args[k[:2]] = k[2:]
    else:
        continue

if args['-i'] == '':
    manual_print()
    raise SystemExit()

try:
    fas, otg, operon, seed, cov, idy, norm, low, up, np, alg, ifl = map(args.get, ['-i', '-r', '-p', '-s', '-c', '-y', '-n', '-l', '-u', '-a', '-A', '-I'])
except:
    manual_print()
    raise SystemExit()


here = os.path.dirname(os.path.abspath(__file__))
sfx = fas.split(os.sep)[-1]


pyc=sys.executable
# mkdir to store the results
os.system('mkdir -p %s_results'%fas)

#######################################################################################
# all-vs-all protein search
#######################################################################################
start = time()

#cmd = 'nohup %s %s/../bin/fast_search.py -p blastp -i %s -d %s -o %s_results/%s.sc -e 1e-5 -s %s -m 5e-2 -a %s -v 100000 > %s_results/log'%(pyc, here, fas, fas, fas, sfx, seed, np, fas)
cmd = 'nohup %s %s/../bin/find_hit.py -p blastp -i %s -d %s -o %s_results/%s.sc -e 1e-5 -s %s -m 5e-2 -a %s -v 100000 > %s_results/log'%(pyc, here, fas, fas, fas, sfx, seed, np, fas)
os.system(cmd)

print 'all to all homologous searching time:', time() - start

#######################################################################################
# identify ortholog, co-ortholog and paralog
#######################################################################################
start = time()

cmd = '%s %s/../bin/find_orth.py -i %s_results/%s.sc -c %s -y %s -n %s -t y > %s_results/%s.opc'
os.system(cmd%(pyc, here, fas, sfx, cov, idy, norm, fas, sfx))

print 'orthomcl algorithm time:', time() - start


#######################################################################################
# use mcl to cluster the genes
#######################################################################################
start = time()

# convert header of sequence to number save memory
flag = 0
id2n = {}
_o = open('%s_results/%s.xyz'%(fas, sfx), 'w')
f = open('%s_results/%s.opc'%(fas, sfx), 'r')
for i in f:
    j = i.split('\t')
    typ, qid, sid, sco = j
    if qid not in id2n:
        id2n[qid] = flag
        flag += 1
    if sid not in id2n:
        id2n[sid] = flag
        flag += 1
    Q = id2n[qid]
    S = id2n[sid]
    _o.write('%d\t%d\t%s'%(Q, S, sco))

f.close()
_o.close()


#cmd = 'cut -f2-4 %s_results/%s.opc > %s_results/%s.xyz'%(fas, sfx, fas, sfx)
#os.system(cmd)

#cmd = 'nohup mcl %s_results/%s.xyz --abc -I 1.5 -o %s_results/%s.grp -te %s > %s_results/log'%(fas, sfx, fas, sfx, np, fas)
cmd = 'nohup %s %s/../bin/find_cluster.py -i %s_results/%s.xyz -a %s -I %s > %s_results/%s.grp'%(pyc, here, fas, sfx, alg, ifl, fas, sfx)
os.system(cmd)

# recover header from number
n2id = {}
while id2n:
    qid, num = id2n.popitem()
    num = str(num)
    n2id[num] = qid

del id2n

_o = open('%s_results/%s.clsr'%(fas, sfx), 'w')
f = open('%s_results/%s.grp'%(fas, sfx), 'r')
for i in f:
    j = i[:-1].split('\t')
    k = map(n2id.get, j)
    #print j, k, n2id
    out = '\t'.join(k)
    _o.write('%s\n'%out)

f.close()
_o.close()

# remove grp file
os.system('rm %s_results/%s.grp'%(fas, sfx))

#print 'use mcl to group protein family time:', time() - start
print 'use %s to group protein family time:'%alg, time() - start


#######################################################################################
# statistics of pan-genome
#######################################################################################
start = time()

#cmd = 'python %s/pangenome.py -i %s -g %s_results/%s.clsr > %s_results/%s.pan'
cmd = '%s %s/pan_genome.py -i %s -g %s_results/%s.clsr > %s_results/%s.pan'%(pyc, here, fas, fas, sfx, fas, sfx)
os.system(cmd)

print 'pan-genome analysis time:', time() - start

#######################################################################################
# use RBH to find and align core genes, then concatenate them to single alignment
#######################################################################################
start = time()

#cmd = 'python %s/rbh2phy.py -f %s -i %s_results/%s.sc > %s_results/%s.aln'
cmd = '%s %s/rbh2phy.py -f %s -i %s_results/%s.sc > %s_results/%s.aln'
os.system(cmd%(pyc, here, fas, fas, sfx, fas, sfx))
# use trimal to remove weak alignment region.
cmd = 'trimal -in %s_results/%s.aln -out %s_results/%s.aln.trim -automated1'
os.system(cmd%(fas, sfx, fas, sfx))
# use fasttree to construct phylotree
cmd = 'fasttree -quiet -wag -gamma -pseudo -spr 4 -mlacc 2 -slownni -no2nd -boot 1000 %s_results/%s.aln.trim > %s_results/%s.nwk'
os.system(cmd%(fas, sfx, fas, sfx))

print 'species tree construction time:', time() - start

#######################################################################################
# cluster the operons if an operon file given
#######################################################################################

if os.path.isfile(operon):
    start = time()
    sfxo = operon.split(os.sep)[-1]
    cmd = '%s %s/operon_cluster.py -g %s_results/%s.clsr -p %s > %s_results/%s.xyz'%(pyc, here, fas, sfx, operon, fas, sfxo)
    #print operon
    #print cmd
    os.system(cmd)

    # use mcl to cluster operon
    #cmd = 'nohup mcl %s_results/%s.xyz --abc -I 1.5 -o %s_results/%s.clsr -te %s'%(fas, sfxo, fas, sfxo, np)
    cmd = 'nohup %s %s/../bin/find_cluster.py -i %s_results/%s.xyz -a %s -I %s > %s_results/%s.clsr'%(pyc, here, fas, sfxo, alg, ifl, fas, sfxo)
    print 'operon clustering time:', time() - start
    os.system(cmd)

os.system('rm -rf %s_results/*_tmp'%fas)
os.system('rm %s_results/*.xyz'%fas)
os.system('rm %s_results/*.aln'%fas)

