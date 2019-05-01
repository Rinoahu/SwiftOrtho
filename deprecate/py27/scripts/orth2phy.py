#!usr/bin/env python
import sys
from Bio import SeqIO
from collections import Counter
import os


# do the core gene find
# python this_script.py -i foo.pep.fsa -g foo.ortholog [-r taxon]
def manual_print():
    print 'This script is used to find orthologs in other species by given a reference, then extract sequences of these orthologs and do a msa, finally, concatenate all the msa.'
    print 'Usage:'
    print '  python this.py -i foo.pep.fsa -g foo.ortholg [-r taxon]'
    print 'Parameters:'
    print ' -i: protein/gene fasta file. The header should be like xxxx|yyyy: xxxx is taxon name and yyyy is unqiue identifier in that taxon'
    print ' -g: orthologs file in the format:\n     O\txxxx|yyyy\tXXXX|YYYY\t...: xxxx/XXXX is taxon name and yyyy/YYYY is unique identifier in that taxon'
    print ' -r: taxonomy name used as reference [optional]'


argv = sys.argv
# recommand parameter:
args = {'-i': '', '-g': '', '-r': ''}

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

if args['-i'] == '' or args['-g'] == '':
    manual_print()
    raise SystemExit()

try:
    fas, orth, taxon = args['-i'], args['-g'], args['-r']
except:
    manual_print()
    raise SystemExit()

# check reference taxon
taxon_ct = Counter()
f = open(fas, 'r')
for i in f:
    if i.startswith('>'):
        j = i[1:-1].split('|')[0]
        taxon_ct[j] += 1

f.close()

# if taxon not specified, then choose the taxon with most genes
taxon_hf = taxon_ct.items()
taxon_hf.sort(key=lambda x:x[1])
taxon_N = len(taxon_hf)
taxon_max = taxon_hf[-1]
taxon = taxon == '' and taxon_max[0] or taxon
#print 'taxon is', taxon, taxon_N

# get the ortholog
# find ortholog in all tax

os.system('mkdir -p ./alns_tmp/')

f = open(orth, 'r')
ortholog = {}
for i in f:
    j = i[:-1].split('\t')
    if len(j) > 3:
        tp, g0, g1 = j[:3]
        t0, t1 = g0.split('|')[0], g1.split('|')[0]
        if tp == 'O' and t0 == taxon:
            try:
                ortholog[g0].append(g1)
            except:
                ortholog[g0] = [g0, g1]
        elif tp == 'O' and t1 == taxon:
            try:
                ortholog[g1].append(g0)
            except:
                ortholog[g1] = [g1, g0]
        else:
            continue

f.close()

taxon_N = max([len(elem) for elem in ortholog.itervalues()])
#print ortholog
orths = []
for i in ortholog.keys():
    j = ortholog[i]
    if len(j) == taxon_N:
        orths.append(j)
    del ortholog[i]

#print orths[:10], taxon_N
orths_set = set()
for i in orths:
    orths_set.update(i)
#print orths_set, 'hello'

seqs_dict = {}
for i in SeqIO.parse(fas, 'fasta'):
    if i.id in orths_set:
        seqs_dict[i.id] = i

#print len(seqs)
#raise SystemExit()

# write the seq to file
orths_N = len(orths)
for i in xrange(orths_N):
    j = orths[i]
    seqs = [seqs_dict[elem] for elem in j]
    _o = open('./alns_tmp/%d.fsa' % i, 'w')
    #_o.write('>%s|%s\n%s\n' % (tax, qid, sq))
    SeqIO.write(seqs, _o, 'fasta')
    _o.close()

#raise SystemExit()
# use the muscle to aln
for i in xrange(orths_N):
    # break
    os.system('muscle -in ./alns_tmp/%d.fsa -out ./alns_tmp/%d.fsa.aln -fasta -quiet' % (i, i))
    #os.system('/home/zhans/tools/tree_tools/trimal/source/trimal -in ./alns_tmp/%d.fsa.aln -out ./alns_tmp/%d.fsa.aln.trim -automated1' % (i, i))


#N = len([elem for elem in os.listdir('./tmpdir') if elem.endswith('.trim')])
# print 'total N', N
tree = {}
for i in xrange(orths_N):
    #seqs = SeqIO.parse('./alns_tmp/%d.fsa.aln.trim' % i, 'fasta')
    seqs = SeqIO.parse('./alns_tmp/%d.fsa.aln' % i, 'fasta')
    for j in seqs:
        taxon = j.id.split('|', 2)[0]
        try:
            tree[taxon].append(str(j.seq))
        except:
            tree[taxon] = [str(j.seq)]


# print 'tree is', tree

#flag = 0
N = len(tree)
L = len(''.join(tree.values()[0]))
#print ' %d %d' % (N, L)
for i in tree:
    hd = '>'+i
    #hd = i
    sq = ''.join(tree[i])
    #print hd, sq
    print hd
    print sq 


os.system('rm -rf alns_tmp')
# combine to single file
# os.system('cd ./tmpdir;python /home/hx/tools/new_genome_project/antismash2.0/antismash-2.0.1/blastout/tree_script/combine.py')
