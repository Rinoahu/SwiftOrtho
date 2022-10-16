#!/usr/bin/env python
# this script is used to recover blast-like hit from the results of nr sequences blast searching
# usage:
# python this.py foo.sc

import sys

try:
    qry = sys.argv[1]
except:
    print('python this.py foo.sc')
    sys.exit()

f = open(qry, 'r')

for i in f:
