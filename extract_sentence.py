#/usr/bin/env python
#-*- coding: utf-8 -*-

""" this program is implemented for extracting sentemces from bilingual corpus and lda training corpus. """

import argparse
import re


parser = argparse.ArgumentParser(description="show this help message")
parser.add_argument("-l", dest="lda",help="lda corpus")
parser.add_argument("-pvt", dest="pvt",help="pvt file")
parser.add_argument("-src_tgt", dest="src_tgt", help="src or tgt file")
parser.add_argument("-w_l",dest="write_l",help="write file for lda")
parser.add_argument("-w_p", dest="write_f",help="write file for pvt")
parser.add_argument("-w_st", dest="write_j", help="write file for src or tgt")
args = parser.parse_args()

p = re.compile(r'^\n')

lda = open(args.lda,"r")
f = open(args.pvt,"r")
j = open(args.src_tgt,"r")
wl = open(args.write_l,"w")
wf = open(args.write_f,"w")
wj = open(args.write_j,"w")

l = []
count = 0
for i in lda:
	if p.search(i):
		l.append(count)
	else:
		wl.write(i)
	count += 1

count_f = 0
for i in f:
	if count_f in l:
		pass
	else:
		wf.write(i)
	count_f += 1

count_j = 0
for i in j:
	if count_j in l:
		pass
	else:
		wj.write(i)
	count_j += 1


