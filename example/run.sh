#!/bin/bash

a=ref.fsa

#python=pypy
#python=python3
python=/home/xiaoh/Downloads/compiler/intel/intelpython27/envs/intelpython/bin/python3.5
#python=python

rm -rf ref.fsa.sc ref.fsa.sc.orth ref.fsa.sc.orth*mcl ref.fsa.sc.orth*apc ref.fsa.sc_tmp ref.fsa_results

#exit 0;

$python ../bin/find_hit.py -p blastp -i $a -d $a -o $a.sc -e 1e-5 -s 1111111 -r aa20

$python ../bin/find_orth.py -i $a.sc -c 0.5 -y 0 > $a.sc.orth

$python ../bin/find_cluster.py -i $a.sc.orth -a apc -I 1.5 > $a.sc.orth_apc

$python ../bin/find_cluster.py -i $a.sc.orth -a mcl -I 1.5 > $a.sc.orth_mcl

echo 'test 1 finished'

$python ../scripts/run_all.py -i $a -p $a\.operon -s 1111111111 -a 2

echo 'test 2 finished'




