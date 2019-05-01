#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# CreateTime: 2016-05-09 15:46:28

import sys
import os
from math import sqrt
from random import random, seed
from rpython.rtyper.lltypesystem.module.ll_math import ll_math_log as log
from rpython.rtyper.lltypesystem.module.ll_math import ll_math_log10 as log10
from rpython.rtyper.lltypesystem.module.ll_math import ll_math_pow as pow
from rpython.rlib import rrandom
from rpython.rlib.rfloat import erfc
from rpython.rtyper.lltypesystem.rffi import r_ushort, r_int
from rpython.rlib.rarithmetic import intmask, r_uint32, r_uint, string_to_int
from rpython.rlib import rfile
from rpython.rlib import rmmap
from rpython.rlib.listsort import TimSort
from rpython.rlib import listsort
from rpython.rlib import rstring
from rpython.rlib.rstruct.runpack import runpack as unpack
from time import time, sleep
from rpython.rlib import rgc


# fnv hash function for 32 bit
def fnv(data, start, end, bit=32):
    if bit == 32:
        a, b, c = 0x811c9dc5, 0x01000193, 0xffffffff
    else:
        a, b, c = 0xcbf29ce484222325, 0x100000001b3, 0xffffffffffffffff

    for i in xrange(start, end):
        s = ord(data[i])
        a ^= s
        a *= b
        a &= c
    return a


# float to sci
def f2s(e):
    if e <= 0:
        E = '0'
    elif 0 < e < 1e-3:
    #if e < 1e-3:
        a = log10(e)
        a -= int(a)
        a = a < 0 and 1 + a or a
        b = pow(10, a)
        s = str(log10(e/b))
        end = max(0, s.find('.'))
        s = s[:end]
        p = str(b)
        end = max(0, p.find('.')+3)
        p = p[:end]
        E = '%se%s'%(p, s)
    else:
        E = str(e)
    return E


uint32 = r_uint32

# the disk base array
class array:

    def __init__(self, fn, dtype='i', mode='r+', shape=()):

        self.fn = fn
        self.dtype = dtype
        self.mode = mode
        self.shape = shape
        if self.dtype == 'i':
            self.stride = 4
        else:
            self.stride = 2

        if self.mode == 'r+':
            f = open(self.fn, 'rb', 1024*1024*256)
            ACCESS=rmmap.ACCESS_READ
        elif mode == 'w+' and shape != ():
            # create the file
            _o = open(self.fn, 'wb', 1024*1024*256)
            ACCESS=rmmap.ACCESS_WRITE
            N = self.stride
            for i in self.shape:
                N *= i
            _o.seek(N-self.stride)
            _o.write('\0'*self.stride)
            _o.close()
            f = open(fn, 'wb', 1024*1024*256)
        else:
            pass

        self.f = f
        self.data = rmmap.mmap(self.f.fileno(), 0, access=ACCESS)
        self.N = self.data.size // self.stride
        if mode == 'r+' and self.shape == ():
            self.shape = (self.N)

    def __len__(self):
        #return self.N
        return self.shape[0]

    def __getitem__(self, i):
        j = self.data.getslice(i*self.stride, self.stride)
        v = unpack(self.dtype, j)
        return v

    def __getslice__(self, i, j):
        return [self.__getitem__(elem) for elem in xrange(i, j)]

    def __setitem__(self, i, v):
        j = pack(self.dtype, v)
        self.data.setslice(i*self.stride, j)

    def __setitem__(self, i, v):
        pass

    def __flush__(self):
        self.f.flush()

    def close(self):
        self.f.close()

# bisect algorithm
# left is to find most left x
def bisect(a, x, l=-1, r=-1, left=True):
    l = l<0 and 0 or l
    r = r<0 and len(a) or r
    if left:
        while r-l > 1:
            m = (l+r)//2
            pv = intmask(a[m])
            if pv < x:
                l = m
            else:
                r = m
    else:
        while r-l > 1:
            m = (l+r)//2
            pv = intmask(a[m])
            if pv > x:
                r = m
            else:
                l = m
    return l


# the counter type
class Counter:
    def __init__(self, seq):
        self.data = {}
        for i in seq:
            try:
                self.data[i] += 1
            except:
                self.data[i] = 0
    def __getitem__(self, i):
        return self.data.get(i, 0)

    def __setitem__(self, i, j):
        self.data[i] = j

    def values(self):
        return self.data.values()

    def keys(self):
        return self.data.keys()


MIN, MED, MAX = 7, 23, 41

random = rrandom.Random(42).random

# swap 2 selected elem in a list
def swap_u(x, i, j):
    x[i], x[j] = x[j], x[i]

# in-place sort
def insort_u(x, l, r, key=lambda x: x):
    for i in xrange(l, r):
        v = x[i]
        pivot = key(v)
        j = i - 1
        while j >= l:
            if key(x[j]) <= pivot:
                break
            x[j + 1] = x[j]
            j -= 1

        x[j + 1] = v


# partition function of quicksort
def partition_u(x, l, r, m, key=lambda x: x):
    t = x[l]
    pivot = key(t)
    i, j = l, r + 1
    while 1:
        i += 1
        while i <= r and key(x[i]) < pivot:
            i += 1
        j -= 1
        while key(x[j]) > pivot:
            j -= 1
        if i > j:
            break
        swap_u(x, i, j)

    swap_u(x, l, j)
    return j

# recursion version of quicksort
def quicksort_u(x, l, r, key=lambda x: x):
    if r <= l:
        return
    else:
        gap = r - l + 1
        if gap < MIN:
            insort_u(x, l, r+1, key)
            return

        elif MIN == gap:
            m = l + gap // 2

        else:
            m = l + int(random() * gap)

        swap_u(x, l, m)
        med = partition_u(x, l, r, m, key)
        quicksort_u(x, l, med-1, key)
        quicksort_u(x, med+1, r, key)

# the main function of qsort
def qsort_u(x, key=lambda x: x):
    quicksort_u(x, 0, len(x)-1, key)



# swap 2 selected elem in a list
def swap(x, i, j):
    x[i], x[j] = x[j], x[i]

# in-place sort
def insort(x, l, r, key=lambda x: x):
    for i in xrange(l, r):
        v = x[i]
        pivot = key(v)
        j = i - 1
        while j >= l:
            if key(x[j]) <= pivot:
                break
            x[j + 1] = x[j]
            j -= 1

        x[j + 1] = v


# partition function of quicksort
def partition(x, l, r, m, key=lambda x: x):
    t = x[l]
    pivot = key(t)
    i, j = l, r + 1
    while 1:
        i += 1
        while i <= r and key(x[i]) < pivot:
            i += 1
        j -= 1
        while key(x[j]) > pivot:
            j -= 1
        if i > j:
            break
        swap(x, i, j)

    swap(x, l, j)
    return j

# recursion version of quicksort
def quicksort(x, l, r, key=lambda x: x):
    if r <= l:
        return
    else:
        gap = r - l + 1
        if gap < MIN:
            insort(x, l, r+1, key)
            return

        elif MIN == gap:
            m = l + gap // 2

        else:
            m = l + int(random() * gap)

        swap(x, l, m)
        med = partition(x, l, r, m, key)
        quicksort(x, l, med-1, key)
        quicksort(x, med+1, r, key)

# the main function of qsort
def qsort(x, key=lambda x: x):
    quicksort(x, 0, len(x)-1, key)



B62 = {('B', 'N'): 3, ('S', 'W'): -3, ('G', 'G'): 6, ('X', 'D'): -1, ('X', 'Z'): -1, ('A', 'N'): -2, ('A', 'Y'): -2, ('W', 'Q'): -2, ('V', 'N'): -3, ('F', 'K'): -3, ('C', 'Z'): -3, ('V', 'X'): -1, ('G', 'E'): -2, ('E', 'D'): 2, ('F', 'Z'): -3, ('W', 'P'): -4, ('H', 'B'): 0, ('I', 'T'): -1, ('F', 'D'): -3, ('E', 'Z'): 4, ('K', 'V'): -2, ('C', 'Y'): -2, ('G', 'D'): -1, ('T', 'N'): 0, ('W', 'W'): 11, ('Y', 'X'): -1, ('E', 'M'): -2, ('S', 'S'): 4, ('X', 'C'): -2, ('X', 'H'): -1, ('K', 'C'): -3, ('E', 'F'): -3, ('Y', 'I'): -1, ('Z', 'P'): -1, ('A', 'K'): -1, ('A', 'B'): -2, ('Q', 'P'): -1, ('F', 'G'): -3, ('D', 'S'): 0, ('Z', 'Q'): 3, ('D', 'X'): -1, ('V', 'T'): 0, ('H', 'P'): -2, ('P', 'V'): -2, ('I', 'Q'): -3, ('Y', 'K'): -2, ('W', 'T'): -2, ('H', 'F'): -1, ('P', 'D'): -1, ('Q', 'R'): 1, ('D', 'Q'): 0, ('K', 'Q'): 1, ('Z', 'S'): 0, ('D', 'F'): -3, ('X', 'G'): -1, ('X', 'L'): -1, ('G', 'Z'): -2, ('V', 'W'): -3, ('T', 'C'): -1, ('A', 'F'): -2, ('T', 'H'): -2, ('A', 'Q'): -1, ('Q', 'T'): -1, ('V', 'F'): -1, ('F', 'C'): -2, ('C', 'R'): -3, ('V', 'P'): -2, ('H', 'T'): -2, ('E', 'L'): -3, ('F', 'R'): -3, ('I', 'G'): -4, ('Z', 'V'): -2, ('Y', 'V'): -1, ('T', 'A'): 0, ('T', 'V'): 0, ('Q', 'V'): -2, ('S', 'K'): 0, ('X', 'K'): -1, ('X', 'P'): -2, ('K', 'K'): 5, ('E', 'N'): 0, ('N', 'T'): 0, ('A', 'H'): -2, ('A', 'C'): 0, ('V', 'S'): -2, ('A', 'Z'): -1, ('M', 'Z'): -1, ('Q', 'H'): 0, ('V', 'B'): -3, ('P', 'X'): -2, ('H', 'S'): -1, ('Q', 'Y'): -1, ('H', 'X'): -1, ('P', 'N'): -2, ('I', 'Y'): -1, ('P', 'G'): -2, ('F', 'N'): -3, ('H', 'N'): 1, ('K', 'H'): -1, ('N', 'W'): -4, ('S', 'Y'): -2, ('W', 'N'): -4, ('D', 'Y'): -3, ('E', 'Q'): 2, ('K', 'Y'): -2, ('S', 'G'): 0, ('X', 'T'): 0, ('Y', 'S'): -2, ('G', 'R'): -2, ('A', 'L'): -1, ('L', 'Z'): -3, ('A', 'G'): 0, ('Y', 'B'): -3, ('T', 'K'): -1, ('T', 'P'): -1, ('M', 'V'): 1, ('Q', 'L'): -2, ('E', 'S'): 0, ('H', 'W'): -2, ('I', 'D'): -3, ('K', 'F'): -3, ('L', 'X'): -1, ('N', 'A'): -2, ('T', 'I'): -1, ('Q', 'N'): 0, ('K', 'W'): -3, ('W', 'B'): -4, ('S', 'C'): -1, ('X', 'S'): 0, ('N', 'B'): 3, ('X', 'X'): -1, ('Z', 'L'): -3, ('Y', 'Y'): 7, ('G', 'V'): -3, ('L', 'V'): 1, ('A', 'R'): -1, ('Z', 'M'): -1, ('M', 'R'): -1, ('N', 'I'): -3, ('D', 'C'): -3, ('P', 'P'): 7, ('D', 'H'): -1, ('Q', 'Q'): 5, ('I', 'V'): 3, ('P', 'F'): -4, ('I', 'A'): -1, ('F', 'F'): 6, ('K', 'T'): -1, ('L', 'T'): -1, ('Q', 'B'): 0, ('S', 'Q'): 0, ('X', 'A'): 0, ('W', 'F'): 1, ('D', 'A'): -2, ('E', 'Y'): -2, ('K', 'A'): -1, ('X', 'W'): -2, ('Q', 'S'): 0, ('A', 'D'): -2, ('L', 'R'): -2, ('T', 'S'): 1, ('A', 'V'): 0, ('T', 'X'): 0, ('M', 'N'): -2, ('Q', 'D'): 0, ('E', 'R'): 0, ('S', 'X'): 0, ('E', 'P'): -1, ('V', 'V'): 4, ('D', 'N'): 1, ('I', 'S'): -2, ('P', 'M'): -2, ('H', 'D'): -1, ('Z', 'B'): 1, ('F', 'B'): -3, ('E', 'E'): 5, ('I', 'L'): 2, ('K', 'N'): 0, ('L', 'P'): -3, ('N', 'L'): -3, ('Y', 'L'): -1, ('T', 'Q'): -1, ('Q', 'F'): -3, ('Z', 'H'): 0, ('S', 'M'): -1, ('X', 'E'): -1, ('W', 'Z'): -3, ('Q', 'W'): -2, ('Z', 'D'): 1, ('G', 'N'): 0, ('L', 'Y'): -1, ('A', 'X'): 0, ('L', 'N'): -3, ('A', 'S'): 1, ('Z', 'E'): 4, ('D', 'T'): -1, ('S', 'T'): 1, ('P', 'Z'): -1, ('P', 'S'): -1, ('V', 'R'): -3, ('D', 'K'): -1, ('P', 'H'): -2, ('H', 'C'): -3, ('Q', 'I'): -3, ('H', 'H'): 8, ('I', 'I'): 4, ('L', 'W'): -2, ('L', 'L'): 4, ('Z', 'G'): -2, ('D', 'R'): -2, ('S', 'I'): -2, ('X', 'I'): -1, ('D', 'I'): -3, ('E', 'A'): -1, ('K', 'I'): -3, ('Q', 'K'): 1, ('G', 'B'): -1, ('T', 'D'): -1, ('A', 'W'): -3, ('Y', 'R'): -2, ('M', 'F'): 0, ('S', 'P'): -1, ('Z', 'I'): -3, ('H', 'Q'): 0, ('E', 'X'): -1, ('F', 'S'): -2, ('I', 'P'): -3, ('E', 'C'): -4, ('H', 'G'): -2, ('P', 'E'): -1, ('Q', 'M'): 0, ('H', 'L'): -3, ('Z', 'Z'): 4, ('L', 'B'): -4, ('T', 'B'): -1, ('L', 'S'): -2, ('L', 'H'): -3, ('N', 'Q'): 0, ('C', 'F'): -2, ('T', 'Y'): -2, ('K', 'G'): -2, ('S', 'E'): 0, ('X', 'M'): -1, ('Y', 'E'): -2, ('W', 'R'): -3, ('V', 'M'): 1, ('N', 'R'): 0, ('C', 'V'): -1, ('G', 'F'): -3, ('F', 'Y'): 3, ('L', 'Q'): -2, ('M', 'Y'): -1, ('A', 'P'): -1, ('D', 'G'): -1, ('C', 'L'): -1, ('L', 'F'): 0, ('D', 'W'): -4, ('M', 'B'): -3, ('S', 'L'): -2, ('P', 'R'): -2, ('P', 'K'): -1, ('Y', 'G'): -3, ('C', 'K'): -3, ('H', 'K'): -1, ('Q', 'A'): -1, ('I', 'F'): 0, ('K', 'D'): -1, ('N', 'C'): -3, ('L', 'D'): -4, ('F', 'V'): -1, ('D', 'Z'): 1, ('S', 'A'): 1, ('X', 'Q'): -1, ('W', 'V'): -3, ('E', 'I'): -3, ('V', 'I'): 3, ('Q', 'C'): -3, ('T', 'G'): -2, ('B', 'P'): -2, ('T', 'L'): -1, ('L', 'M'): 2, ('A', 'T'): 0, ('C', 'H'): -3, ('C', 'A'): 0, ('Y', 'Z'): -2, ('S', 'Z'): 0, ('P', 'Y'): -3, ('S', 'H'): -1, ('B', 'Q'): 0, ('H', 'Y'): 2, ('I', 'X'): -1, ('E', 'K'): 1, ('C', 'G'): -3, ('I', 'C'): -1, ('Q', 'E'): 2, ('K', 'R'): 2, ('Z', 'R'): 0, ('T', 'E'): -1, ('B', 'R'): -1, ('L', 'K'): -2, ('M', 'W'): -1, ('N', 'Y'): -2, ('B', 'S'): 0, ('E', 'B'): 1, ('Y', 'M'): -1, ('V', 'E'): -2, ('N', 'Z'): 0, ('Z', 'T'): -1, ('Y', 'D'): -3, ('B', 'T'): -1, ('F', 'Q'): -3, ('G', 'Y'): -3, ('L', 'I'): 2, ('M', 'Q'): 0, ('R', 'A'): -1, ('C', 'D'): -3, ('S', 'V'): -2, ('D', 'D'): 6, ('S', 'D'): 0, ('P', 'C'): -3, ('G', 'X'): -1, ('R', 'B'): -1, ('C', 'C'): 9, ('W', 'K'): -3, ('I', 'N'): -3, ('B', 'V'): -3, ('K', 'L'): -2, ('M', 'X'): -1, ('N', 'K'): 0, ('L', 'G'): -4, ('M', 'S'): -1, ('R', 'C'): -3, ('X', 'B'): -1, ('Z', 'W'): -3, ('D', 'B'): 4, ('B', 'W'): -4, ('X', 'Y'): -1, ('R', 'D'): -2, ('V', 'A'): 0, ('W', 'I'): -3, ('B', 'X'): -1, ('T', 'T'): 5, ('F', 'M'): 0, ('L', 'E'): -3, ('M', 'M'): 5, ('R', 'E'): 0, ('W', 'H'): -2, ('S', 'R'): -1, ('E', 'W'): -3, ('P', 'Q'): -1, ('B', 'Y'): -3, ('H', 'A'): -2, ('N', 'D'): 1, ('E', 'H'): 0, ('R', 'F'): -3, ('I', 'K'): -3, ('K', 'Z'): 1, ('N', 'E'): 0, ('T', 'M'): -1, ('B', 'Z'): 1, ('T', 'R'): -1, ('M', 'T'): -1, ('G', 'S'): 0, ('L', 'C'): -1, ('R', 'G'): -2, ('N', 'H'): 1, ('X', 'F'): -1, ('N', 'F'): -3, ('Y', 'Q'): -1, ('N', 'P'): -2, ('R', 'H'): 0, ('W', 'M'): -1, ('C', 'N'): -3, ('V', 'L'): 1, ('F', 'I'): 0, ('G', 'Q'): -2, ('L', 'A'): -1, ('M', 'I'): 1, ('R', 'I'): -3, ('W', 'L'): -2, ('Q', 'G'): -2, ('S', 'N'): 1, ('D', 'L'): -4, ('F', 'X'): -1, ('I', 'R'): -3, ('P', 'B'): -2, ('C', 'M'): -1, ('H', 'E'): 0, ('Y', 'W'): 2, ('G', 'P'): -2, ('W', 'C'): -2, ('Z', 'K'): 1, ('M', 'P'): -2, ('N', 'S'): 1, ('G', 'W'): -2, ('M', 'K'): -1, ('R', 'K'): 2, ('D', 'E'): 2, ('K', 'E'): 1, ('R', 'L'): -2, ('A', 'I'): -1, ('V', 'Y'): -1, ('W', 'A'): -3, ('Y', 'F'): 3, ('T', 'W'): -2, ('V', 'H'): -3, ('F', 'E'): -3, ('M', 'E'): -2, ('R', 'M'): -1, ('C', 'X'): -2, ('E', 'T'): -1, ('H', 'R'): 0, ('P', 'I'): -3, ('F', 'T'): -2, ('B', 'A'): -2, ('Z', 'N'): 0, ('H', 'I'): -3, ('G', 'T'): -2, ('I', 'H'): -3, ('R', 'N'): 0, ('C', 'W'): -2, ('W', 'G'): -2, ('K', 'B'): 0, ('Y', 'H'): 2, ('B', 'B'): 4, ('T', 'Z'): -1, ('M', 'L'): 2, ('G', 'K'): -2, ('M', 'G'): -3, ('K', 'S'): 0, ('E', 'V'): -2, ('X', 'N'): -1, ('N', 'N'): 6, ('B', 'C'): -3, ('V', 'K'): -2, ('N', 'X'): -1, ('R', 'P'): -2, ('A', 'M'): -1, ('W', 'E'): -3, ('V', 'Z'): -2, ('F', 'W'): 1, ('B', 'D'): 4, ('Z', 'A'): -1, ('V', 'D'): -3, ('F', 'A'): -2, ('G', 'I'): -4, ('M', 'A'): -1, ('R', 'Q'): 1, ('C', 'T'): -1, ('W', 'D'): -4, ('H', 'V'): -3, ('S', 'F'): -2, ('P', 'T'): -1, ('F', 'P'): -4, ('I', 'Z'): -3, ('B', 'E'): 1, ('C', 'E'): -4, ('H', 'M'): -2, ('I', 'E'): -3, ('G', 'H'): -2, ('R', 'R'): 5, ('K', 'P'): -1, ('C', 'S'): -1, ('B', 'F'): -3, ('Z', 'C'): -3, ('D', 'V'): -3, ('M', 'H'): -2, ('M', 'C'): -1, ('R', 'S'): -1, ('D', 'M'): -3, ('X', 'R'): -1, ('K', 'M'): -1, ('B', 'G'): -1, ('C', 'I'): -1, ('V', 'G'): -3, ('R', 'T'): -1, ('A', 'A'): 4, ('V', 'Q'): -2, ('W', 'Y'): 2, ('Y', 'N'): -2, ('B', 'H'): 0, ('C', 'B'): -3, ('G', 'M'): -3, ('C', 'P'): -3, ('W', 'X'): -2, ('H', 'Z'): 0, ('S', 'B'): 0, ('E', 'G'): -2, ('I', 'W'): -3, ('P', 'A'): -1, ('F', 'L'): 0, ('B', 'I'): -3, ('Z', 'F'): -3, ('G', 'L'): -4, ('R', 'V'): -3, ('T', 'F'): -2, ('C', 'Q'): -3, ('Y', 'P'): -3, ('M', 'D'): -3, ('G', 'C'): -3, ('R', 'W'): -3, ('Y', 'A'): -2, ('X', 'V'): -1, ('N', 'V'): -3, ('B', 'K'): 0, ('Z', 'X'): -1, ('N', 'G'): 0, ('V', 'C'): -1, ('R', 'X'): -1, ('A', 'E'): -1, ('Q', 'X'): -1, ('N', 'M'): -2, ('B', 'L'): -4, ('Z', 'Y'): -2, ('D', 'P'): -1, ('G', 'A'): 0, ('R', 'Y'): -2, ('P', 'W'): -4, ('Y', 'C'): -2, ('P', 'L'): -3, ('F', 'H'): -1, ('I', 'B'): -3, ('B', 'M'): -3, ('I', 'M'): 1, ('Y', 'T'): -2, ('R', 'Z'): 0, ('K', 'X'): -1, ('Q', 'Z'): 3, ('W', 'S'): -3}
def dict2mat(B62):
    b62 = [[-4] * 256 for elem in xrange(256)]
    for a, b in B62:
        score = B62[(a, b)]
        A = [a, a.lower()]
        B = [b, b.lower()]
        for i in A:
            for j in B:
                b62[ord(i)][ord(j)] = score
                b62[ord(j)][ord(i)] = score

    return b62

b62 = dict2mat(B62)


# read file line by line
def readline(f, chk=256):
    c = ''
    while 1:
        a = f.read(1024*1024*chk)
        if not a:
            break

        b = a.split('\n')
        b[0] = c + b[0]
        for line in b[:-1]:
            yield '%s\n' % line

        c = b[-1]

# fasta parse
def parse(f):
    head, seq = '', []
    for i in readline(f):
        n = len(i) - 1
        n = n > 0 and n or 0
        if i.startswith('>'):
            if len(seq) > 0:
                yield head, ''.join(seq)
            head = i[1:n]
            seq = []
        else:
            seq.append(i[:n])

    if len(seq) > 0:
        yield head, ''.join(seq)


# reduce aa table
aa_nr = 'KREDQN,C,G,H,ILV,M,F,Y,W,P,STA'

# generate aa nr table
# AA is groupped aa acid
def generate_nr_tbl(gaa=aa_nr):
    aa = gaa.upper().split(',')
    aa_nr_tbl = [0] * 256
    flag = 0
    #for c0 in aa_nr:
    for c0 in aa:
        flag += 1
        for c1 in c0:
            c1l = c1.lower()
            aa_nr_tbl[ord(c1)] = flag
            aa_nr_tbl[ord(c1l)] = flag

    return aa_nr_tbl

aa_nr_tbl = generate_nr_tbl(aa_nr)


# kmer2num
def k2n(s, st=-1, ed=-1, scale=-1, code=aa_nr_tbl):
    if st == -1 and ed == -1:
        st, ed = 0, len(s)
    if scale == -1:
        scale = Max(code) + 1

    Scale = 1
    n = 0
    for i in xrange(st, ed):
        c = ord(s[i])
        n += code[c] * Scale
        Scale *= scale

    return n

# seq num
def seq2n(s, k=5, scale=-1, code=aa_nr_tbl):
    if scale == -1:
        scale = Max(code) + 1

    n = k2n(s, 0, k)
    Scale = int(pow(scale, k-1))
    yield n
    for i in xrange(k, len(s)):
        c = ord(s[i])
        n //= scale
        n += code[c] * Scale
        yield n


nr_aa = ['X', 'AST', 'CFILMVY', 'DN', 'EQ', 'G', 'H', 'KR', 'P', 'W']
# num2seq
def n2s(n, k=-1, scale=-1, code=nr_aa):
    if scale == -1:
        scale = len(code)
    if k == -1:
        #k = int(log(n) / log(10)) + 1
        k = int(log10(n))
    s = []
    for i in xrange(k):
        j = n % scale
        s.append(code[j])
        n //= scale

    return s

# mutiple spaced seeds
def spseeds_fnv(seq, step=1, scale=-1, code=aa_nr_tbl, max_weight=-1, ssps='11111111,11101011101,1011001011101', mod=1):
    if scale == -1:
        scale = Max(code) + 1
    spaces = ssps.split(',')
    # get max weight
    mw = max_weight < 0 and Max([ssp.count('1') for ssp in spaces]) or max_weight
    # set offset for the k2n
    base = int(pow(scale, mw))
    L, S = len(seq), len(spaces)
    for s in xrange(S):
        space = spaces[s]
        k = len(space)
        for i in xrange(0, L-k+1, step):
            seg, n, b, c = True, 0x811c9dc5, 0x01000193, 0xffffffff
            for j in xrange(k):
                ij = i+j
                if seq[ij] == 'x' or seq[ij] == 'X':
                    seg = False
                    break
                elif space[j] != '0':
                    char = ord(seq[i+j])
                    n ^= char
                    n *= b
                    n &= c
                else:
                    continue
            n ^= s
            n *= b
            n &= c
            if seg:
                yield n%mod, i

# normal spseeds function
def spseeds_nm(seq, step=1, scale=-1, code=aa_nr_tbl, max_weight=-1, ssps='11111111,11101011101,1011001011101', mod=1):
    if scale == -1:
        scale = Max(code) + 1
    spaces = ssps.split(',')
    # get max weight
    mw = max_weight < 0 and Max([ssp.count('1') for ssp in spaces]) or max_weight
    # set offset for the k2n
    base = int(pow(scale, mw))
    L, S = len(seq), len(spaces)
    for s in xrange(S):
        space = spaces[s]
        k = len(space)
        for i in xrange(0, L-k+1, step):
            seg, n, Scale = True, 0, 1
            for j in xrange(k):
                ij = i+j
                if seq[ij] == 'x' or seq[ij] == 'X':
                    seg = False
                    break
                if space[j] != '0':
                    #c = ord(seq[i+j])
                    c = ord(seq[i+j])
                    n += code[c] * Scale
                    Scale *= scale
            n += base * s
            #print 'seed num is', S, 'k is', k, space, n, scale
            if seg:
                yield n%mod, i

spseeds = spseeds_fnv

# convert protein sequence to array
def kmer_slw(seqs, k=5, scale=-1, code=aa_nr_tbl):
    if scale == -1:
        scale = Max(code) + 1
    tbl = {}
    idx = 0
    total = 0
    for hd, sq in seqs:
        for i in xrange(len(sq)-k+1):
            key = k2n(sq, i, i + k, scale, code)
            try:
                tbl[key].append([idx, i])
            except:
                tbl[key] = [[idx, i]]
            if total % 1000000 == 0:
                print total, len(tbl)
            total += 1

        idx += 1
    return tbl


# kmer sequence
def kmer(seqs, k=5, scale=-1, code=aa_nr_tbl):
    if scale == -1:
        scale = Max(code) + 1

    tbl = [[] for elem in xrange(int(pow(10, k)))]
    #tbl = {}
    idx = 0
    total = 0
    for hd, sq in seqs:
        i = 0
        for key in seq2n(sq, k):
            tbl[key].extend([r_uint32(idx), r_uint32(i)])

            i += 1
            if total % 10000000 == 0:
                print total, len(tbl)
            total += 1

        idx += 1
    return tbl


# reverse sequence
def reverse(seq):
    N = len(seq)
    end = N - 1
    for i in xrange(N//2):
        seq[i], seq[end-i] = seq[end-i], seq[i]
    #return seq


def reverse_char(seq):
    N = len(seq)
    end = N - 1
    for i in xrange(N//2):
        seq[i], seq[end-i] = seq[end-i], seq[i]


# longest increase sequencing
def lis(seq, key = lambda x: x[0]):
    if len(seq) < 2:
        return seq

    N = len(seq)
    M, P = [-1] * N, [-1] * N
    L, M[0] = 1, 0

    # Looping over the sequence starting from the second element
    for i in xrange(1, N):
        lower = 0
        upper = L
        if key(seq[M[upper-1]]) < key(seq[i]):
            j = upper
        else:
            # actual binary search loop
            while upper - lower > 1:
                mid = (upper + lower) // 2
                if key(seq[M[mid-1]]) < key(seq[i]):
                    lower = mid
                else:
                    upper = mid
            j = lower    # this will also set the default value to 0
        P[i] = M[j-1]
        if j == L or key(seq[i]) < key(seq[M[j]]):
            M[j] = i
            L = max(L, j+1)
    # Building the result: [seq[M[L-1]], seq[P[M[L-1]]], seq[P[P[M[L-1]]]], ...]
    result = []
    pos = M[L-1]
    for elem in xrange(L):
        result.append(seq[pos])
        pos = P[pos]

    reverse(result)
    return result


# the sum, mean and std of an array/list
def sum(x):
    flag = 0
    for i in x:
        flag += intmask(i)
    return flag


# get the mean of an array
mean = lambda x: 1. * intmask(sum(x)) / len(x)


# return the mu and sd
def sd(x):
    mu = mean(x)
    return sqrt(sum([pow(elem-mu, 2) for elem in x]) * 1. / len(x))


# get mean and mu
def get_mu_sd(x, m=0):
    N = 1
    mu = 0.
    for j in x:
        i = intmask(j)
        if i > m:
            mu += i
            N += 1
    mu /= N
    sd = 0.
    for j in x:
        i = intmask(j)
        if i > m:
            sd += pow(i-mu, 2)
    sd = sqrt(sd/N)
    return mu, sd

def get_mu_sd0(x):
    N = len(x)
    mu = mean(x)
    sd = 0
    for i in x:
        sd += pow(intmask(i)-mu, 2.)
    sd = sqrt(sd / N)
    return mu, sd



# convert number to string
def pack(dtype, value):
    t = dtype.lower()
    if t == 'h':
        n = 2
    elif t == 'i':
        n = 4
    elif t == 'l':
        n = 8
    else:
        n = int(dtype)

    val = intmask(value)
    string = [''] * n
    for i in xrange(n):
        string[i] = chr(val & 0xFF)
        #val = val >> 8
        val >>= 8

    return ''.join(string)


# unpack of pack
def upack(dtype, string):
    t = dtype.lower()
    if t == 'h':
        n = 2
    elif t == 'i':
        n = 4
    elif t == 'l':
        n = 8
    else:
        n = int(dtype)

    val = 0
    for i in xrange(n-1, -1, -1):
        val <<= 8
        val += ord(string[i])

    return val


# lis for 2 neighbor val list
def lis2(seq):
    n = len(seq)
    sst = [intmask(seq[elem]) for elem in xrange(1, n, 2)]
    if len(seq) <= 2:
        return seq

    N = len(sst)
    M, P = [-1] * N, [-1] * N
    L, M[0] = 1, 0
    # Looping over the sequence starting from the second element
    for i in xrange(1, N):
        lower = 0
        upper = L
        if sst[M[upper-1]] < sst[i]:
            j = upper
        else:
            # actual binary search loop
            while upper - lower > 1:
                mid = (upper + lower) // 2
                if sst[M[mid-1]] < sst[i]:
                    lower = mid
                else:
                    upper = mid
            # this will also set the default value to 0
            j = lower
        P[i] = M[j-1]
        if j == L or sst[i] < sst[M[j]]:
            M[j] = i
            L = max(L, j+1)
    # Building the result: [seq[M[L-1]], seq[P[M[L-1]]], seq[P[P[M[L-1]]]], ...]
    result = []
    pos = M[L-1]
    for elem in xrange(L):
        result.extend([seq[pos*2+1], seq[pos*2]])
        pos = P[pos]

    reverse(result)
    return result


# max of seq
def Max(L):
    n = len(L)
    if n <= 1:
        return L[0]
    else:
        flag = L[0]
        for i in xrange(1, n):
            flag = L[i] > flag and L[i] or flag
    return flag

# min of seq
def Min(L):
    n = len(L)
    if n <= 1:
        return L[0]
    else:
        flag = L[0]
        for i in xrange(1, n):
            flag = L[i] < flag and L[i] or flag
    return flag


# convert raw score to ncbi bit
def score2bit(score, gap=True):
    if gap:
        bit = (.267 * score + 3.1941832122778293) / 0.69314718055994529
    else:
        bit = (.309 * score + 1.9589953886039688) / 0.69314718055994529
    return int(bit)


# convert ncbi bit to raw score
def bit2score(bit, gap=True):
    if gap:
        score = (bit * 0.69314718055994529 - 3.1941832122778293) / .267
    else:
        score = (bit * 0.69314718055994529 - 1.9589953886039688) / .309
    return score

# calculate e value
# D is size of database
# sqi, sqj are the sequence
# bit is the ncbi bit score of sqi and sqj
bit2e = lambda D, sqi, sqj, bit: D * len(sqi) * len(sqj) * pow(2, -bit)


# smith waterman gotoh
def swat(s0, s1, mch=2, go=-11, ge=-1, mat=b62, kbound=32):
    l0 = len(s0) + 1
    l1 = len(s1) + 1
    score = [[0] * l0 for elem in xrange(l1)]
    trace = [['*'] * l0 for elem in xrange(l1)]
    for i in xrange(1, l0):
        #score[0][i] = go + i * ge
        trace[0][i] = '-'
    for i in xrange(1, l1):
        #score[i][0] = go + i * ge
        trace[i][0] = '|'

    i_max = j_max = maxscore = 0
    for i in xrange(1, l1):
        for j in xrange(1, l0):
            I = score[i][j-1] + (trace[i][j-1] == '-' and ge or go)
            c1, c0 = s1[i-1], s0[j-1]
            M = score[i-1][j-1] + mat[ord(c1)][ord(c0)]
            D = score[i-1][j] + (trace[i-1][j] == '|' and ge or go)
            B = Max([0, I, M, D])
            score[i][j] = B

            if B > maxscore:
                i_max = i
                j_max = j
                maxscore = B

            if B == 0:
                trace[i][j] = '*'
            elif B == I:
                trace[i][j] = '-'
            elif B == D:
                trace[i][j] = '|'
            else:
                trace[i][j] = '\\'

    al0, al1 = [], []
    i, j = i_max, j_max
    while i > 0 or j > 0:
        if trace[i][j] == '\\':
            al0.append(s0[j-1])
            al1.append(s1[i-1])
            i -= 1
            j -= 1
        elif trace[i][j] == '-':
            al0.append(s0[j - 1])
            al1.append('-')
            j -= 1
        elif trace[i][j] == '|':
            al1.append(s1[i - 1])
            al0.append('-')
            i -= 1
        elif trace[i][j] == '*':
            break
        else:
            print 'next'
    al0.reverse()
    al1.reverse()
    return ''.join(al0), ''.join(al1), j, j_max, i, i_max, score2bit(maxscore)


# use kmer base lis to find rough start end of qry and target
def klis2(s0, s1, k=7, scale=-1, code=aa_nr_tbl, kbound=12):
    l0, l1 = len(s0), len(s1)
    if scale == -1:
        scale = Max(code) + 1
    kdict = {}
    j = 0
    for kn in seq2n(s1, k, scale, code):
        #kdict[kn].append(j)
        try:
            kdict[kn].append(j)
        except:
            kdict[kn] = [j]

        j += 1
    pairs = {}
    total = i = mkey = flag = 0
    for kn in seq2n(s0, k, scale, code):
        for j in kdict.get(kn, []):
        #for j in kdict[kn]:
            key = (i-j) // kbound
            #key = (i-j+l1) // kbound
            try:
                pairs[key].extend([i, j])
            except:
                pairs[key] = [i, j]
            pairs[key].extend([i, j])
            if len(pairs[key]) > flag:
                mkey, flag = key, len(pairs[key])
            #total += 1
        i+=1
    res = lis2(pairs[mkey])
    a, c = res[:2] 
    end = max(len(res)-2, 0)
    b, d = res[end:]
    return [a, b, c, d]


# find max hit diag
def Diag(s0, s1, k=4, scale=-1, code=aa_nr_tbl, kbound=32):
    scale = scale == -1 and Max(code) + 1 or scale
    kdict = {}
    j = 0
    for kn in seq2n(s1, k, scale, code):
        try:
            kdict[kn].append(j)
        except:
            kdict[kn] = [j]

        j += 1

    width = kbound
    pairs = {}
    i = 0
    mkey = flag = 0
    for kn in seq2n(s0, k, scale, code):
        for j in kdict.get(kn, []):
            key = (i-j) // width
            try:
                pairs[key] += 1
            except:
                pairs[key] = 1

            if pairs[key] > flag:
                mkey, flag = key, pairs[key]

        i+=1

    if mkey < 0:
        i, j = 0, -mkey * width
    elif mkey > 0:
        i, j = mkey * width, 0
    else:
        i = j = 0
    return mkey, width, i, j



# kswat from given start of qry and ref sequence
# qst: start of qry, sst: start of ref
def max_idx(a, n=4):
    idx = 0
    for i in xrange(1, n):
        if a[idx] < a[i]:
            idx = i
    return idx


# opt: best/full output the best/full local alignment
# can be used to do reverse alignment of two sequence
def kswat_st0(S0, S1, qst=0, qed=-1, sst=0, sed=-1, mch=2, go=-11, ge=-1, mat=b62, kbound=16, score=[[]], trace=[[]], opt='best', al0=[], al1=[]):
    if len(S0) < len(S1):
        s0, s1, swap = S0, S1, False
    else:
        s0, s1, swap = S1, S0, True

    qst = min(max(qst, 0), len(S0))
    qed = qed < 0 and len(S0) or qed
    sst = min(max(sst, 0), len(S1))
    sed = sed < 0 and len(S1) or sed

    qsp = qst < qed and 1 or -1
    ssp = sst < sed and 1 or -1

    l0 = abs(qed - qst) + 1
    l1 = abs(sed - sst) + 1

    score = score != [[]] and score or [[0] * l0 for elem in xrange(l1)]
    trace = trace != [[]] and trace or [['*'] * l0 for elem in xrange(l1)]
    for i in xrange(1, l0):
        score[0][i] = 0
        trace[0][i] = '-'
    for i in xrange(1, l1):
        score[i][0] = 0
        trace[i][0] = '|'
        start, end = max(0, i-kbound-1), min(i+kbound+1, l0-1)
        trace[i][start], trace[i][end] = '|', '-'
        score[i][start] = score[i][end] = 0


    i_max = j_max = maxscore = 0
    for i in xrange(1, l1):
        start, end = max(1, i-kbound), min(i+kbound, l0)
        for j in xrange(start, end):
            I = score[i][j-1] + (trace[i][j-1] == '-' and ge or go)
            c1, c0 = s1[(i-1)*ssp+sst], s0[(j-1)*qsp+qst]
            M = score[i-1][j-1] + mat[ord(c1)][ord(c0)]
            D = score[i-1][j] + (trace[i-1][j] == '|' and ge or go)
            B = Max([0, I, M, D])
            score[i][j] = B
            if B > maxscore:
                i_max, j_max, maxscore = i, j, B

            if B == M:
                trace[i][j] = '\\'
            elif B == I:
                trace[i][j] = '-'
            elif B == D:
                trace[i][j] = '|'
            else:
                trace[i][j] = '*'

    if opt == 'full':
        i, j = l1-1, l0-1
    else:
        i, j = i_max, j_max

    while i > 0 or j > 0:
        tij = trace[i][j]
        if tij == '\\':
            al0.append(s0[(j-1)*qsp+qst])
            al1.append(s1[(i-1)*ssp+sst])
            i -= 1
            j -= 1

        elif tij == '-':
            al0.append(s0[(j-1)*qsp+qst])
            al1.append('-')
            j -= 1

        elif tij == '|':
            al1.append(s1[(i-1)*ssp+sst])
            al0.append('-')
            i -= 1

        else:
            if opt == 'best':
                break
            else:
                al0.append(s0[(j-1)*qsp+qst])
                al1.append(s1[(i-1)*qsp+sst])
                i -= 1
                j -= 1

    if qst < qed:
        al0.reverse()
    else:
        i, i_max = i_max, i
    if sst < sed:
        al1.reverse()
    else:
        j, j_max = j_max, j

    AL = len(al0)
    idy = mis = gap = 0
    op = -1
    for idx in xrange(AL):
        if al0[idx] == al1[idx]:
            idy += 1.
        else:
            mis += 1
        if al0[idx] == '-' and op != 0:
            gap += 1
            op = 0
        elif al1[idx] == '-' and op != 1:
            gap += 1
            op = 1
        else:
            op = -1

    idy *= (100./AL)

    if swap:
        return idy, AL, mis, gap, i*ssp+sst, i_max*ssp+sst, j*qsp+qst, j_max*qsp+qst, score2bit(maxscore)
    else:
        return idy, AL, mis, gap, j*qsp+qst, j_max*qsp+qst, i*qsp+sst, i_max*qsp+sst, score2bit(maxscore)

def kswat_st(S0, S1, qst=0, qed=-1, sst=0, sed=-1, mch=2, go=-11, ge=-1, mat=b62, kbound=16, score=[[]], trace=[[]], opt='best', al0=[], al1=[]):

    qst = min(max(qst, 0), len(S0))
    qed = qed < 0 and len(S0) or qed
    sst = min(max(sst, 0), len(S1))
    sed = sed < 0 and len(S1) or sed

    if abs(qed-qst) < abs(sed-sst):
        s0, s1, swap = S0, S1, False
    else:
        s0, s1, swap = S1, S0, True
        qst, qed, sst, sed = sst, sed, qst, qed
        al0, al1 = al1, al0


    qsp = qst < qed and 1 or -1
    ssp = sst < sed and 1 or -1

    l0 = abs(qed - qst) + 1
    l1 = abs(sed - sst) + 1

    score = score != [[]] and score or [[0] * l0 for elem in xrange(l1)]
    trace = trace != [[]] and trace or [['*'] * l0 for elem in xrange(l1)]
    for i in xrange(1, l0):
        #score[0][i] = go + i * ge
        score[0][i] = 0
        trace[0][i] = '-'
    for i in xrange(1, l1):
        #score[i][0] = go + i * ge
        score[i][0] = 0
        trace[i][0] = '|'
        start, end = max(0, i-kbound-1), min(i+kbound+1, l0-1)
        trace[i][start], trace[i][end] = '|', '-'
        score[i][start] = score[i][end] = 0


    i_max = j_max = maxscore = 0
    for i in xrange(1, l1):
        start, end = max(1, i-kbound), min(i+kbound, l0)
        for j in xrange(start, end):
            I = score[i][j-1] + (trace[i][j-1] == '-' and ge or go)
            c1, c0 = s1[(i-1)*ssp+sst], s0[(j-1)*qsp+qst]
            M = score[i-1][j-1] + mat[ord(c1)][ord(c0)]
            D = score[i-1][j] + (trace[i-1][j] == '|' and ge or go)
            B = Max([0, I, M, D])
            score[i][j] = B
            if B > maxscore:
                i_max, j_max, maxscore = i, j, B

            if B == M:
                trace[i][j] = '\\'
            elif B == I:
                trace[i][j] = '-'
            elif B == D:
                trace[i][j] = '|'
            else:
                trace[i][j] = '*'

    if opt == 'full':
        i, j = l1-1, l0-1
    else:
        i, j = i_max, j_max

    while i > 0 or j > 0:
        tij = trace[i][j]
        if tij == '\\':
            al0.append(s0[(j-1)*qsp+qst])
            al1.append(s1[(i-1)*ssp+sst])
            i -= 1
            j -= 1

        elif tij == '-':
            al0.append(s0[(j-1)*qsp+qst])
            al1.append('-')
            j -= 1

        elif tij == '|':
            al1.append(s1[(i-1)*ssp+sst])
            al0.append('-')
            i -= 1

        else:
            if opt == 'best':
                break
            else:
                al0.append(s0[(j-1)*qsp+qst])
                al1.append(s1[(i-1)*qsp+sst])
                i -= 1
                j -= 1

    if qst < qed:
        al0.reverse()
    else:
        i, i_max = i_max, i
    if sst < sed:
        al1.reverse()
    else:
        j, j_max = j_max, j

    AL = len(al0)
    idy = mis = gap = 0
    op = -1
    for idx in xrange(AL):
        if al0[idx] == al1[idx]:
            idy += 1.
        else:
            mis += 1
        if al0[idx] == '-' and op != 0:
            gap += 1
            op = 0
        elif al1[idx] == '-' and op != 1:
            gap += 1
            op = 1
        else:
            op = -1

    idy *= (100./AL)

    if swap:
        return idy, AL, mis, gap, i*ssp+sst, i_max*ssp+sst, j*qsp+qst, j_max*qsp+qst, score2bit(maxscore)
    else:
        return idy, AL, mis, gap, j*qsp+qst, j_max*qsp+qst, i*qsp+sst, i_max*qsp+sst, score2bit(maxscore)



# kswat for long sequences
def kswat_st_long(sqi, sqj, qi, qj, chk=4096, score=[[]], trace=[[]], al0=[], al1=[]):
    li, lj = len(sqi), len(sqj)
    j = qj

    score = score != [[]] and score or [[0] * 4100 for elem in xrange(4100)]
    trace = trace != [[]] and trace or [['*'] * 4100 for elem in xrange(4100)]

    for i in xrange(qi, li, chk):
        i, ied, j, jed = max(0, i), max(0, i+chk), max(0, j), max(0, j+chk)
        idy, aln, mis, gap, qst, qed, sst, sed, bit = kswat_st(sqi[i:ied], sqj[j:jed], qst=0, sst=0, score=score, trace=trace, al0=al0, al1=al1)
        qst += i
        qed += i
        sst += j
        sed += j
        yield idy, aln, mis, gap, qst, qed, sst, sed, bit
        del al0[:]
        del al1[:]
        j += chk


# mix of ungap align + kswat
# input: seqi seqj [0, 0, qst0, sst0, qed0, sed0, qst1, sst1, qed1, sst1, ... qstN, sstN, qedN, sstN, -1, -1]
def ungap_kswat_st(s0, s1, starts, score_mat, trace_mat, dropX=30):
    if not starts:
        return kswat_st(s0, s1)
    N = len(starts)
    Aln0, Aln1 = [], []
    first = 0
    last = N - 4
    for i in xrange(0, N-2, 2):
        qst, sst, qed, sed = start[i:i+4]
        if (i//2) % 2 == 0:
            if first < i < last:
                idy, aln, mis, gap, qst, qed, sst, sed, bit, alni, alnj = kswat_st(sqi, sqj, qst=qi, sst=qj, score=score_mat, trace=trace_mat, opt='full')

            elif i == first:
                idy, aln, mis, gap, qst, qed, sst, sed, bit, alni, alnj = kswat_st(sqi, sqj, qst=qi, sst=qj, score=score_mat, trace=trace_mat, opt='right')
            else:
                idy, aln, mis, gap, qst, qed, sst, sed, bit, alni, alnj = kswat_st(sqi, sqj, qst=qi, sst=qj, score=score_mat, trace=trace_mat, opt='left')

        else:
            Aln0.append(s0[qst: qed])
            Aln1.append(s0[sst: sed])


# index fasta file, record the position of >
def index0(f):
    fasta = rmmap.mmap(f.fileno(), 0, access=rmmap.ACCESS_READ)
    N = fasta.size
    idx = []
    for i in xrange(N):
        if fasta.getitem(i) == '>':
            j = i - 1
            if i == 0 or fasta.getitem(j) == '\n':
                idx.append(i)
    return fasta, N, idx

def index(f):
    fasta = rmmap.mmap(f.fileno(), 0, access=rmmap.ACCESS_READ)
    N = fasta.size
    idx = [0]
    for i in xrange(N):
        if fasta.getitem(i) == '>':
            j = i - 1
            if i > 0 or fasta.getitem(j) == '\n':
                idx.append(i)
    return fasta, N, idx



# get fasta file by mmap and index
class Fasta:

    def __init__(self, f):
        self.fasta, self.end, self.idx = index(f)
        self.N = len(self.idx)

    def __getitem__(self, X = 0):
        x = intmask(X)
        if x < 0:
            x += self.N
        if 0 <= x < self.N:
            if x == self.N - 1:
                start, end = self.idx[x], self.end
            else:
                start, end = self.idx[x], self.idx[x+1]
            fsa = self.fasta.getslice(start, end - start).split('\n')
            hd, sq = fsa[0][1:], ''.join(fsa[1:])
            return [hd, sq]
        else:
            return ['', '']

    def __len__(self):
        return self.N

    def get_hdseq(self, i):
        return self.hdseqs[i-self.offset]

    # build on disk
    def build_msav(self, space='11111111', nr=aa_nr, step=1, scale=-1, start=-1, end=-1, memory=True, ht=-1):

        self.code = generate_nr_tbl(nr)
        if scale==-1:
            scale = Max(self.code) + 1

        self.scale = scale
        self.nr = nr
        self.space = space
        self.mink = Min([len(elem) for elem in self.space.split(',')])
        # get max weight of spaced seed
        self.mw = Max([sp.count('1') for sp in self.space.split(',')])
        # get number of spaces seed
        self.nssp = self.space.count(',') + 1
        self.min = 25
        self.offset = start
        self.offend = end + 1
        self.step = step
        bins = min(int(pow(self.scale, self.mw))*self.nssp*5, 128*1024*1024)
        NC = ht < 1 and bins or ht
        self.NC = NC
        self.start = [r_uint32(0)] * self.NC
        start = min(max(0, start), self.N)
        end = min(end < 0 and self.N or end, self.N)

        M = end - start
        # get the start of all sequences
        self.soas = [r_uint32(0)] * (M + 1)

        for i in xrange(start, end):
            hd, sq = self[i]
            j = i - start
            self.soas[j+1] = r_uint32(intmask(self.soas[j]) + len(sq))
            for key, idx in spseeds(sq, step=step, scale=self.scale, code=self.code, max_weight=self.mw, ssps=self.space, mod=self.NC):
                v0 = self.start[key]
                self.start[key] = r_uint32(intmask(v0) + 1)

        u, sd = get_mu_sd(self.start)
        mu = int(u + 2 * sd)
        self.threshold = mu
        rgc.collect()

        for i in xrange(1, NC):
            v0, v1 = self.start[i-1], self.start[i]
            self.start[i] = uint32(intmask(v0) + intmask(v1))

        self.locus = [r_uint32(0)] * self.start[NC-1]
        for i in xrange(start, end):
            hd, sq = self[i]
            j = i - start
            off = intmask(self.soas[j])
            for key, idx in spseeds(sq, step=step, scale=self.scale, code=self.code, max_weight=self.mw, ssps=self.space, mod=self.NC):
                v0 = self.start[key]
                self.start[key] = uint32(intmask(v0) - 1)
                k = self.start[key]
                self.locus[k] = uint32(idx + off)

        # add features
        _o = open('/tmp/tmp.bin', 'wb')
        _o.write('\0')
        _o.close()
        f = open('/tmp/tmp.bin', 'rb')
        self.start_dsk = rmmap.mmap(f.fileno(), 0, access=rmmap.ACCESS_READ)

        # read the sequence to the memory
        self.memory = memory
        self.L = len(self.locus) - 1
        if self.memory:
            self.hdseqs = [self[elem] for elem in xrange(self.offset, self.offend)]


    # build database on disk
    def makedb(self, name, space='11111111,11100110100101010', nr=aa_nr, step=-1, scale=-1, start=-1, end=-1, memory=False, ht=-1, chk=500000):
        # build small part of the sequence
        N = self.N
        Start = 0 if start == -1 else start
        End = N if end == -1 else end
        for i in xrange(Start, End, chk):
            start, end = i, min(i+chk, End)
            idx = i // chk
            self.build_msav(space=space, nr=nr, step=step, start=start, end=end, ht=ht)
            if memory == False:
                self.write(name + '.%d'%idx, True)
            yield start, end

    # write db to disk
    def write(self, name, add=True):

        chk = 1024 * 128
        _o = open(name+'.idx', 'wb')
        N = len(self.locus)
        for i in xrange(0, N, chk):
            j = min(i+chk, N)
            out = ''.join([pack('i', self.locus[elem]) for elem in xrange(i, j)])
            _o.write(out)
            del out
            rgc.collect()
        del self.locus[:]
        _o.close()

        _o = open(name+'.soas', 'wb')
        N = len(self.soas)
        for i in xrange(0, N, chk):
            j = min(i+chk, N)
            out = ''.join([pack('i', self.soas[elem]) for elem in xrange(i, j)])
            _o.write(out)
            del out
            rgc.collect()

        del self.soas[:]
        _o.close()

        _o = open(name+'.bin', 'wb')
        N = len(self.start)
        for i in xrange(0, N, chk):
            j = min(i+chk, N)
            out = ''.join([pack('i', self.start[elem]) for elem in xrange(i, j)])
            _o.write(out)
            del out
            rgc.collect()

        # write addition parameters to the database
        if add:
            offset = self.offset
            offend = self.offend
            ssp = self.space
            nr = self.nr
            ksz = self.mw
            nc = self.NC
            thr = int(self.threshold)
            out = '%d;%d;%d;%d;%d;%s;%s'%(offset, offend, ksz, thr, nc, ssp, nr)
            _o.write(out+chr(len(out)))

        _o.close()

        del self.start[:]
        rgc.collect()

    # load the database
    def load(self, name, code=aa_nr_tbl, memory=False):

        self.memory = memory
        f0 = open(name + '.idx', 'rb')
        locus_dsk = rmmap.mmap(f0.fileno(), 0, access=rmmap.ACCESS_READ)
        self.L = locus_dsk.size // 4 - 1
        if memory:
            N = locus_dsk.size
            self.locus = [r_uint32(0)] * (N // 4)
            for i in xrange(0, N, 4):
                j = locus_dsk.getslice(i, 4)
                v = uint32(unpack('i', j))
                self.locus[i//4] = v

            f0.close()
            f0 = open(name + '.idx', 'rb')
            locus_dsk = rmmap.mmap(f0.fileno(), 4, access=rmmap.ACCESS_READ)
            self.locus_dsk = locus_dsk
            rgc.collect()
        else:
            self.locus = [uint32(0)]
            self.locus_dsk = locus_dsk

        f1 = open(name + '.bin', 'rb')
        start_dsk = rmmap.mmap(f1.fileno(), 0, access=rmmap.ACCESS_READ)

        # load parameters from bin
        N = start_dsk.size
        M = ord(start_dsk.getitem(N-1))
        start = N - M - 1
        start = max(start, 0)

        para = start_dsk.getslice(start, M)
        offset, offend, mw, thr, nc, space, nr = para.split(';')
        self.offset = int(offset)
        self.offend = int(offend)
        self.NC = int(nc)

        # read the sequence to the memory
        self.hdseqs = [self[elem] for elem in xrange(self.offset, self.offend)]

        # get max weight
        self.mw = int(mw)
        self.threshold = int(thr)
        self.space = space
        # get the min kmer
        self.mink = Min([len(elem) for elem in self.space.split(',')])

        self.nr = nr
        self.code = generate_nr_tbl(nr)
        self.min = 25
        self.scale = Max(self.code) + 1

        self.nssp = self.space.count(',') + 1

        if memory:
            N = start_dsk.size
            self.start = [uint32(0)] * (N // 4)
            for i in xrange(0, N, 4):
                j = start_dsk.getslice(i, 4)
                v = unpack('i', j)
                self.start[i//4] = uint32(v)

            rgc.collect()
            self.start_dsk = start_dsk
            rgc.collect()
        else:
            self.start = [uint32(0)]
            self.start_dsk = start_dsk

        f2 = open(name + '.soas', 'rb')
        soas_dsk = rmmap.mmap(f2.fileno(), 0, access=rmmap.ACCESS_READ)
        N = soas_dsk.size
        self.soas = [r_uint32(0)] * (N // 4)

        for i in xrange(0, N, 4):
            j = soas_dsk.getslice(i, 4)
            v = uint32(unpack('i', j))
            self.soas[i//4] = v
        del soas_dsk
        rgc.collect()
        f2.close()

        if not memory:
            self.f0 = f0
            self.f1 = f1
            self.f2 = f1

    # close the fasta class
    def close(self):
        if not self.memory:
            self.f0.close()
            self.f1.close()
            self.f2.close()

    # ungapped alignment from ncbi blast
    def ungap(self, qseq, sseq, Qst, Sst, qlo=-1, qup=-1, slo=-1, sup=-1, dropX=30):
        qlo = qlo > -1 and qlo or 0
        slo = slo > -1 and slo or 0
        ql, sl = len(qseq), len(sseq)
        qup = qup > -1 and qup or ql
        sup = sup > -1 and sup or sl
        off = max(max(qlo - Qst, slo - Sst), 0)
        Qst += off
        Sst += off
        qst, sst = Qst, Sst
        score, max_score, max_qed, max_sed = 0, 0, qst, sst
        flag = 0
        while qlo < qst < qup and slo < sst < sup:
            flag += 1
            c0, c1 = ord(qseq[qst]), ord(sseq[sst])
            score += b62[c0][c1]
            if score > max_score:
                max_score, max_qed, max_sed = score, qst, sst
            elif score + dropX < max_score:
                break
            else:
                pass
            qst += 1
            sst += 1

        qst, sst = Qst-1, Sst-1
        score, max_qst, max_sst = max_score, qst, sst
        while qup > qst > qlo and sup > sst > slo:
            flag += 1
            c0, c1 = ord(qseq[qst]), ord(sseq[sst])
            score += b62[c0][c1]
            if score > max_score:
                max_score, max_qst, max_sst = score, qst, sst
            elif score + dropX < max_score:
                break
            else:
                pass
            qst -= 1
            sst -= 1

        return max_score, max_qst, max_qed, max_sst, max_sed, flag

    # get ungapped alignment score
    def get_ungap_scores(self, qseq, sseq, loc1):
        qst, sst = loc1[0]
        scores, qst, qed, sst, sed, flag = self.ungap(qseq, sseq, qst, sst)
        x0, y0 = qst, sst
        x, y = qed, sed
        for qst, sst in loc1[1:]:
            score, qst, qed, sst, sed, flag1 = self.ungap(qseq, sseq, qst, sst, qlo=x, slo=y)
            flag += flag1
            x, y = qed, sed
            scores += score

        return scores, flag, x0, y0, x, y


    def get_loc_mem(self, i):
        x = intmask(self.locus[i])
        idx = bisect(self.soas, x)
        return idx+self.offset, x-intmask(self.soas[idx])

    # bin is used to store the range of location of kmer
    # get the start and end of the bin of kmer
    def get_bin(self, i):
        L = self.L
        bins = self.start_dsk.getslice(i*4, 8)
        try:
            start, end = unpack('ii', bins)
        except:
            start, end = unpack('i', bins[:4]), L

        start = max(start, 0)
        end = min(max(end, 0), L)
        return start, end

    def get_bin_mem(self, i):
        L = self.L
        i = i > 0 and i or 0
        try:
            st, ed = self.start[i:i+2]
        except:
            st = ed = self.start[i]
        start, end = intmask(st), intmask(ed)
        start = max(start, 0)
        end = min(max(end, 0), L)

        return start, end


    # get start for query and ref sequences from loc
    def guess_start(self, loc):
        N = len(loc)
        dist = 0
        for qst, sst in loc:
            dist += (sst - qst)
        dist /= N
        if dist > 0:
            return 0, dist
        else:
            return -dist, 0

    # find the hits of sequence and filter by count
    def find_msav(self, seq, kbound=12):
        # get the hit
        s2a = [elem for elem in spseeds(seq, scale=self.scale, code=self.code, max_weight=self.mw, ssps=self.space, mod=self.NC)]

        # find the threshold of hit
        hist = [0] * len(s2a)
        for i, qst in s2a:
            start, end = self.get_bin_mem(i)
            count = end - start
            hist[qst] += count > 0 and count or 0

        hist_c = hist[:]
        TimSort(hist).sort()
        cum = thr = 0
        for i in hist:
            if cum > self.threshold:
                break
            else:
                cum += i
                thr = i
        for i in xrange(len(s2a)):
            hist_c[i] = hist_c[i] > thr and -1 or 1

        hits = {}
        for i, qst in s2a:
            start, end = self.get_bin_mem(i)
            count = end - start
            if hist_c[qst] > 0:
                for j in xrange(start, end):
                    hd, sst = self.get_loc_mem(j)
                    k0 = (qst-sst) // kbound // 2
                    try:
                        hits[(hd, k0)].append([qst, sst])
                    except:
                        hits[(hd, k0)] = [[qst, sst]]

        flag = 0
        Hits, Scores = {}, {}
        for hit in hits:
            hd, k0= hit
            sseq = self[hd][1]
            loc0 = hits[hit]
            qsort(loc0, key=lambda x:x[0])
            loc1 = lis(loc0, key=lambda x:x[1])
            score, flag1, qst, sst, qed, sed = self.get_ungap_scores(seq, sseq, loc1)
            flag += flag1
            if score < self.min:
                continue
            if hd not in Scores:
                Hits[hd] = loc1
                Scores[hd] = score
            elif score > Scores[hd]:
                Hits[hd] = loc1
                Scores[hd] = score
            else:
                continue

        score_hits = []
        for hd in Scores:
            loc, score = Hits[hd], Scores[hd]
            qi, qj = self.guess_start(loc)
            score_hits.append([hd, score, qi, qj])

        qsort(score_hits, key=lambda x: -x[1])
        return score_hits

    # find hit by dsk
    # get the index and hit location of a seq
    def get_loc(self, i):
        x = unpack('i', self.locus_dsk.getslice(i*4, 4))
        idx = bisect(self.soas, x)
        return idx+self.offset, x-intmask(self.soas[idx])


    def get_locs(self, start, count):
        bins = self.locus_dsk.getslice(start*4, count*4)
        for j in xrange(0, len(bins), 4):
            x = unpack('i', bins[j:j+4])
            idx = bisect(self.soas, x)
            yield idx+self.offset, x-intmask(self.soas[idx])

    def get_locs_m(self, start, end):
        for i in xrange(start, end):
            x = intmask(self.locus[i])
            idx = bisect(self.soas, x)
            yield idx+self.offset, x-intmask(self.soas[idx])

    # find the hit by given a sequence
    def find_msav_m(self, seq, kbound=1, sort=True):
        # get kmer blosum62 score
        ql = len(seq)
        kscs, sc = [0] * (ql-self.mink+1), 0
        for i in xrange(self.mink):
            c = ord(seq[i])
            sc += b62[c][c]
        kscs[0] = sc
        for i in xrange(1, ql-self.mink+1):
            c0, c1 = ord(seq[i-1]), ord(seq[i-1+self.mink])
            sc = kscs[i-1] - b62[c0][c0] + b62[c1][c1]
            kscs[i] = sc

        s2a = [elem for elem in spseeds(seq, scale=self.scale, code=self.code, max_weight=self.mw, ssps=self.space, mod=self.NC)]
        hist = [[kscs[qst], qst, 0] for qst in xrange(len(kscs))]
        get_bin = self.get_bin_mem
        for i, qst in s2a:
            start, end = get_bin(i)
            count = end - start
            hist[qst][2] += (count > 0 and count or 0)

        thr = self.threshold * len(seq)
        qsort(hist, key=lambda x:-x[0])
        hist_c = [-1] * ql
        cum = 0
        for i in xrange(len(hist)):
            if cum > thr:
                break
            else:
                ksc, qst, ct = hist[i]
                cum += ct
                hist_c[qst] = 1

        hits = {}
        flag = 0
        for i, qst in s2a:
            start, end = get_bin(i)
            count = end - start
            if hist_c[qst] > 0:
                locs = self.get_locs_m(start, end)
                for hd, sst in locs:
                    k0 = (qst-sst) // kbound
                    try:
                        hits[(hd, k0)].append([qst, sst])
                    except:
                        hits[(hd, k0)] = [[qst, sst]]

        flag = 0
        Hits, Scores = {}, {}

        for hit in hits:
            if len(hit) < 2:
                continue
            hd, k0 = hit
            sseq = self.get_hdseq(hd)[1]
            loc0 = hits[hit]
            qsort(loc0, key=lambda x:x[0])
            loc1 = lis(loc0, key=lambda x:x[1])
            score, flag1, qst, sst, qed, sed = self.get_ungap_scores(seq, sseq, loc1)
            flag += flag1
            if score < self.min:
                continue
            if hd not in Scores or score > Scores[hd]:
                Hits[hd] = [[qst, sst], [qed, sed]]
                Scores[hd] = score
            else:
                continue

        score_hits = []
        for hd in Scores:
            loc, score = Hits[hd], Scores[hd]
            qi, qj = self.guess_start(loc)
            score_hits.append([hd, score, qi, qj])

        if sort:
            qsort(score_hits, key=lambda x: -x[1])

        return score_hits

    # find the hit by given a sequence
    def find_msav_dsk(self, seq, kbound=1, sort=True):

        # get kmer blosum62 score
        ql = len(seq)
        kscs, sc = [0] * (ql-self.mink+1), 0
        for i in xrange(self.mink):
            c = ord(seq[i])
            sc += b62[c][c]
        kscs[0] = sc
        for i in xrange(1, ql-self.mink+1):
            c0, c1 = ord(seq[i-1]), ord(seq[i-1+self.mink])
            sc = kscs[i-1] - b62[c0][c0] + b62[c1][c1]
            kscs[i] = sc

        s2a = [elem for elem in spseeds(seq, scale=self.scale, code=self.code, max_weight=self.mw, ssps=self.space, mod=self.NC)]
        # find the threshold of hit
        hist = [[kscs[qst], qst, 0] for qst in xrange(len(kscs))]
        get_bin = self.get_bin
        for i, qst in s2a:
            start, end = get_bin(i)
            count = end - start
            hist[qst][2] += (count > 0 and count or 0)

        thr = self.threshold * len(seq)
        qsort(hist, key=lambda x:-x[0])
        hist_c = [-1] * ql
        cum = 0
        for i in xrange(len(hist)):
            if cum > thr:
                break
            else:
                ksc, qst, ct = hist[i]
                cum += ct
                hist_c[qst] = 1

        hits = {}
        for i, qst in s2a:
            start, end = get_bin(i)
            count = end - start
            if hist_c[qst] > 0:
                locs = self.get_locs(start, count)
                for hd, sst in locs:
                    k0 = (qst-sst) // kbound
                    try:
                        hits[(hd, k0)].append([qst, sst])
                    except:
                        hits[(hd, k0)] = [[qst, sst]]


        flag = 0
        Hits, Scores = {}, {}
        for hit in hits:
            if len(hit) < 2:
                continue
            hd, k0 = hit
            sseq = self.get_hdseq(hd)[1]
            loc0 = hits[hit]
            qsort(loc0, key=lambda x:x[0])
            loc1 = lis(loc0, key=lambda x:x[1])
            score, flag1, qst, sst, qed, sed = self.get_ungap_scores(seq, sseq, loc1)
            flag += flag1
            if score < self.min:
                continue
            if hd not in Scores or score > Scores[hd]:
                Hits[hd] = [[qst, sst], [qed, sed]]
                Scores[hd] = score
            else:
                continue

        score_hits = []
        for hd in Scores:
            loc, score = Hits[hd], Scores[hd]
            qi, qj = self.guess_start(loc)
            score_hits.append([hd, score, qi, qj])

        if sort:
            qsort(score_hits, key=lambda x: -x[1])
        return score_hits


# make database
def makedb(ref, space='11111111', nr=aa_nr, step=1, ht=-1, chk=500000):
    f1 = open(ref, 'r', 1073741824)
    DB = Fasta(f1)
    # get the size of qry and db
    for i, j in DB.makedb(ref, space=space, nr=nr, step=step, ht=ht, chk=chk):
        i


# the kolmogorov complexity algorithm used for filtering
def kolmogorov(S):
    s = S
    n = len(S), c, l, i, k, k_max, stop
    c = l = k = k_max = 1
    i = stop = 0
    while stop == 0:
        if s[i + k] != s[l + k]:
            if k > k_max:
                k_max = k
            i += 1

            if i == l:
                c += 1
                l += k_max
                if l + 1 >= n:
                    stop = 1
                else:
                    i = 0
                    k = 1
                    k_max = 1
            else:
                k = 1

        else:
            k += 1
            if l + k > n:
                c += 1
                stop = 1

    b = n / log2(n)

    return c / b

# calculate the entropy of a string
def entropy(S):
    s = S.upper()
    n = len(s) * 1.
    counts = Counter(s)
    for c in s:
        counts[c] += 1.
    ent = 0
    for j in counts.values():
        #j = counts[i]
        freq = j / n
        ent -= freq * log(freq)

    ent /= log(2)

    return counts, ent


# seg program like methd to filter low complexity region
def seg(S, minent=2.2, window=12.):
    s = S.upper()
    log2 = log(2)
    n = len(s)
    winsize = int(window)
    counts, ent = entropy(s[: winsize])
    #mask = [0 for elem in xrange(n)]
    mask = [0] * n

    if ent < minent:
        mask[0] = 1

    for i in xrange(1, n - winsize + 1):
        # update the counts
        pre_chr = s[i - 1]
        cur_chr = s[i + 11]

        if pre_chr == cur_chr:
            mask[i] = mask[i-1]
            continue

        # be careful here
        pre_count = counts[pre_chr] # pre must > 1
        counts[pre_chr] -= 1
        cur_count = counts[cur_chr] # cur may = 0
        counts[cur_chr] += 1
        a, b = pre_count / window, counts[pre_chr] / window
        ent += (b != 0 and (a * log(a) - b * log(b)) / log2 or a * log(a) / log2)
        a, b = cur_count / window, counts[cur_chr] / window
        ent += (a != 0 and (a * log(a) - b * log(b)) / log2 or -b * log(b) / log2)

        if ent < minent:
            mask[i] = 1

    Nws = max(0, n-winsize)
    if mask[Nws] == 1:
        for i in xrange(Nws, n):
            mask[i] = 1

    Xs = 'x' * winsize
    output = ''
    st = 0

    for i in xrange(n):
        if st >= n:
            break
        if mask[st] == 0:
            output += s[st]
            st += 1
        else:
            output += Xs
            st += 12

    output = output[: n]
    masked = ''
    st, ed = 0, 0

    for i in xrange(n):
        if ed >= n:
            break
        elif output[i] == 'x':
            curst, cured = max(i - 8, 0), min(i + 9, n)
            if st <= curst <= ed:
                ed = cured
            else:
                masked += 'x' * (ed - st)
                masked += output[ed: curst]
                st, ed = curst, cured
    masked += 'x' * (ed - st)
    masked += output[ed: ]

    return output[: n], masked

# extract batch of sequences
def get_batch_sequences(seqs, st, ed, chk=500000):
    for i in xrange(st, ed, chk):
        output = [seqs[j] for j in xrange(i, min(ed, i+chk))]
        if output:
            yield output

# blast like search
#     qry: query sequences
#     ref: database
#     exp: expect value
#       v: max hit to report
#     thr: threshold to filter high frequency kmer
#   st/ed: start and end of query sequences used to search
# ref_idx: index for large formated database


# blastp without built index
def blastp(qry, ref, expect=1e-5, v=500, max_miss=1e-3, st=-1, ed=-1, rst=-1, red=-1, thr=-1, flt='T', ref_idx='', memory=True, ssd='', nr='', step=4, ht=-1, chk=100000, tmpdir='./tmpdir'):

    max_miss = max(max_miss, 1e-3)
    f0 = open(qry, 'rb')
    seqs = Fasta(f0)
    N = len(seqs)
    f1 = open(ref, 'rb')

    DB = Fasta(f1)
    # get the size of db
    D = len(DB)
    st = min(max(0, st), N)
    ed = min(ed < 0 and D or ed, N)
    score_mat = [[0]*4100 for elem in xrange(4100)]
    trace_mat = [['*']*4100 for elem in xrange(4100)]
    ST = st
    nst = 0
    KDB = {}
    #_o = open('/tmp/%d.tmp.array'%st, 'wb')
    _o = open('%s/%d.tmp.array'%(tmpdir, st), 'wb')

    for a, b in DB.makedb(ref, space=ssd, nr=nr, step=step, memory=True, start=rst, end=red, ht=ht, chk=chk):

        DB.threshold = thr < 1 and DB.threshold or thr
        for i in xrange(st, ed):
            hdi, Sqi = seqs[i]
            if flt == 'T':
                sqi, mask = seg(Sqi)
            else:
                sqi, mask = Sqi, Sqi

            li = len(sqi)
            hits = DB.find_msav_m(sqi.upper(), sort=False)
            hbs = []
            for hit in hits:
                hbs.extend([pack('i', uint32(elem)) for elem in hit])

            _o.write(''.join(hbs))
            lhb = len(hbs) * 4
            try:
                KDB[i].extend([nst, lhb])
            except:
                KDB[i] = [nst, lhb]
            nst += lhb

        rgc.collect()
        DB.close()
        f1.close()

    _o.close()

    #fa = open('/tmp/%d.tmp.array'%st, 'rb')
    fa = open('%s/%d.tmp.array'%(tmpdir, st), 'rb')
    ACCESS=rmmap.ACCESS_READ
    array = rmmap.mmap(fa.fileno(), 0, access=ACCESS)
    for i in xrange(st, ed):
        if i not in KDB:
            continue
        # get the query sequence
        hdi, Sqi = seqs[i]
        if flt == 'T':
            sqi, mask = seg(Sqi)
        else:
            sqi, mask = Sqi, Sqi

        li = len(sqi)
        starts = KDB[i]
        hits = []

        lsts = len(starts)
        for j in xrange(0, lsts, 2):
            nst, lhb = starts[j:j+2]
            string = array.getslice(nst, lhb)
            for k in xrange(0, lhb, 16):
                a0, a1, a2, a3 = unpack('iiii', string[k:k+16])
                hits.append([a0, a1, a2, a3])

        qsort(hits, key=lambda x:-x[1])
        mmiss = len(hits) * max_miss + 1
        mmiss = max(mmiss, 100./mmiss)
        mmiss = min(max(mmiss, 10), 120)
        Unmch = unmch = 0
        bv = 0
        breakpoint = 0
        dp_count = 0
        vmax = max(100, max(v+100, v*1.1))

        m8s = []
        for hit in hits[:vmax]:
            j, sc, qi, qj = hit
            hdj, sqj = DB[j]
            lj = len(sqj)
            hi, hj = hdi.split(' ')[0], hdj.split(' ')[0]
            aln0, aln1 = [], []
            if len(sqi) < 4096 and len(sqj) < 4096:
                idy, aln, mis, gap, qst, qed, sst, sed, bit = kswat_st(sqi, sqj, qst=qi, sst=qj, score=score_mat, trace=trace_mat, al0=aln0, al1=aln1)
                e = bit2e(D, sqi, sqj, bit)
                if e <= expect:
                    #m8 = i, j, li, lj, hi, hj, idy, aln, mis, gap, qst+1, qed, sst+1, sed, e, bit, sc, breakpoint, len(hits), intmask(DB.threshold)
                    m8 = i, j, li, lj, hi, hj, idy, aln, mis, gap, qst+1, qed, sst+1, sed, e, bit, sc, breakpoint, len(hits), intmask(DB.threshold), hdj
                    m8s.append(m8)
                    unmch = 0
                    bv += 1
                else:
                    unmch += 1
                    Unmch += 1
            else:
                flag = 1
                for m9 in kswat_st_long(sqi, sqj, qi, qj, score=score_mat,trace=trace_mat, al0=aln0, al1=aln1):
                    idy, aln, mis, gap, qst, qed, sst, sed, bit = m9
                    e = bit2e(D, sqi, sqj, bit)
                    if e <= expect:
                        #m8 = i, j, li, lj, hi, hj, idy, aln, mis, gap, qst+1, qed, sst+1, sed, e, bit, sc, breakpoint, len(hits), intmask(DB.threshold)
                        m8 = i, j, li, lj, hi, hj, idy, aln, mis, gap, qst+1, qed, sst+1, sed, e, bit, sc, breakpoint, len(hits), intmask(DB.threshold), hdj
                        m8s.append(m8)

                        flag = 0
                        bv += 1
                if flag == 1:
                    unmch += 1
                    Unmch += 1
                else:
                    unmch = 0

            if unmch >= mmiss or bv >= v + mmiss:
                break

            breakpoint += 1

        qsort_u(m8s, key = lambda x: -x[15])
        for m8 in m8s[:max(0, v)]:
            yield m8

        if i % 1000 == 0:
            del hits
            rgc.collect()

    fa.close()

    #os.system('rm /tmp/%d.tmp.array'%st)
    os.system('rm %s/%d.tmp.array'%(tmpdir, st))

    f0.close()


# print the manual
def manual_print():
    print 'Usage:'
    print '  fsearch -p blastp -i qry.fsa -d db.fsa'
    print 'Parameters:'
    print '  -p: program'
    print '  -i: query sequences in fasta format'
    print '  -l: start index of query sequences'
    print '  -u: end index of query sequences'
    print '  -L: start index of reference'
    print '  -U: end index of reference'
    print '  -d: ref database'
    print '  -D: index of ref, if this parameter is specified, only this part of formatted ref will be searched against'
    print '  -o: output file'
    print '  -O: write mode of output file. w: overwrite, a: append'
    print '  -s: spaced seed in format: 1111,1110,1001.. etc'
    print '  -r: reduced amino acid alphabet in format: AST,CFILMVY,DN,EQ,G,H,KR,P,W'
    print '  -v: number of hits to show'
    print '  -e: expect value'
    print '  -m: max ratio of pseudo hits that will trigger stop'
    print '  -j: distance between start sites of two neighbor seeds, greater will reduce the size of database'
    print '  -t: filter high frequency kmers whose counts > t'
    print '  -F: Filter query sequence'
    print '  -M: bucket size of hash table, reduce this parameter will reduce memory usage but decrease sensitivity'
    print '  -c: chunck size of reference. default is 50K which mean 50K sequences from reference will be used as database'
    print '  -T: tmpdir to store tmp file. default ./tmpdir'


def entry_point(argv):

    # 1x6
    seeds = '111111'
    #seeds = '1111111,11010010111,110100010001011'
    #seeds = '11111111,11101011011'
    # 8x6 weight 6
    #seeds = '1110010011,110010100011,10100100001011,10100010010101,11000001010011,1100010000001011,110100000001000011,1101000000000001000101'
    # 16x12 weight 12
    #seeds = '111101011101111,111011001100101111,1111001001010001001111,111100101000010010010111'
    aa_nr = 'AST,CFILMVY,DN,EQ,G,H,KR,P,W'
    #aa_nr = 'A,KR,EDNQ,C,G,H,ILVM,FYW,P,ST'
    #aa_nr = 'KREDQN,C,G,H,ILV,M,F,Y,W,P,STA'
    #aa_nr = 'G,P,IV,FYW,A,LM,EQRK,ND,HS,T,C'
    # recommand parameter:
    args = {'-p':'', '-v':'500', '-s':seeds, '-i':'', '-d':'', '-e':'1e-3', '-l':'-1', '-u':'-1', '-m':'1e-3', '-t':'-1', '-r':aa_nr, '-j':'4', '-F':'T', '-o':'', '-D':'', '-O':'wb', '-L':'-1', '-U':'-1', '-M':'-1', '-c':'50000', '-T':'./tmpdir'}
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


    if args['-p'] not in ['blastp']:
        manual_print()
        return 0
    elif args['-p'] == 'blastp':
        if args['-i'] == '' or args['-d'] == '':
            manual_print()
            return 0
    else:
        pass

    # get parameters and start the program
    if args['-p'] == 'blastp':

        try:
            qry, ref, exp, bv, start, end, rstart, rend, miss, thr, step, flt, outfile, ref_idx, wrt, ht, chk, tmpdir = args['-i'], args['-d'], float(args['-e']), int(args['-v']), int(args['-l']), int(args['-u']), int(args['-L']), int(args['-U']), float(args['-m']), int(args['-t']), int(args['-j']), args['-F'], args['-o'], args['-D'], args['-O'], int(args['-M']), int(args['-c']), args['-T']
        except:
            print 'blastp'
            manual_print()
            return 0

        ssd, nr = args['-s'], args['-r']
        wrt = wrt in 'wa' and wrt or 'w'
        _o = outfile and open(outfile, wrt) or open('/dev/null', 'rb')
        abp = os.path.abspath(ref)
        abp = abp[:max(0, abp.rfind(os.sep))]
        fn = ref[max(0, ref.rfind(os.sep)+1):]
        ssd, nr = args['-s'], args['-r']
        wrt = wrt in 'wa' and wrt or 'w'
        m8s = []
        for hit in blastp(qry, ref, expect=exp, v=bv, max_miss=miss, st=start, ed=end, rst=rstart, red=rend, thr=thr, flt=flt, ref_idx=ref_idx, ssd=ssd, nr=nr, step=step, ht=ht, chk=chk, tmpdir=tmpdir):
            #i, j, li, lj, hi, hj, idy, aln, mis, gap, qst, qed, sst, sed, e, bit, seed, bv, vl, thr = hit
            i, j, li, lj, hi, hj, idy, aln, mis, gap, qst, qed, sst, sed, e, bit, seed, bv, vl, thr, desc = hit
            if e <= exp:
                Idy = str(idy)
                End = max(0, Idy.find('.')+3)
                Idy = Idy[:End]
                E = f2s(e)
                #m8 = '%s\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%s\t%f\t%d\t%d\t%d\t%d\t%f\t%d\t%d\t%d\n'%(hi, hj, Idy, aln, mis, gap, qst, qed, sst, sed, E, bit, i, j, li, lj, seed, bv, vl, thr)
                #m8 = '%s\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%s\t%f\t%d\t%d\n'%(hi, hj, Idy, aln, mis, gap, qst, qed, sst, sed, E, bit, li, lj)
                m8 = '%s\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%s\t%f\t%d\t%d\t%s\n'%(hi, hj, Idy, aln, mis, gap, qst, qed, sst, sed, E, bit, li, lj, desc)
                m8s.append(m8)
                if outfile:
                    if len(m8s) >= 10000:
                        _o.write(''.join(m8s))
                        m8s = []
                    else:
                        continue
                else:
                    print m8.strip()

        if outfile and len(m8s) > 0:
            _o.write(''.join(m8s))

        _o.close()


    else:
        manual_print()
        return 0

    return 0

def target(*args):
    return entry_point, None

if __name__ == "__main__":

    entry_point(sys.argv)
