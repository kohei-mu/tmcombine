#!/usr/bin/env python
# -*- coding: utf-8 -*-

# preprocess script for GibbsLDA++
# make training corpus for topic model

import sys,re,argparse
import multiprocessing as mp
from functools import partial
from collections import defaultdict
argvs = sys.argv

stoplist = [word.strip() for word in open("stopwords", "r")]
digit = re.compile(r'\d+')

def write(doc,proc,p):
	length = len(doc)
	ini = length * p / proc
	fin = length * (p+1) /proc
	result = []
	documents = doc[ini:fin]
	texts = [ [word for word in document.lower().split() if word not in stoplist] for document in documents]
	
    freq = defaultdict(int)
	for text in texts:
		for word in text:
			freq[word] += 1
	texts = [[word for word in text if freq[word] > 1 and not digit.search(word) ] for text in texts]

	for text in texts:
		if argvs[4] == "1":
			if len(text) > 0:
				result.append(" ".join(text)+"\n")
		elif argvs[4] == "0":
			result.append(" ".join(text)+"\n")

	return result

def main(proc):
	documents = open(argvs[1],"r").readlines()
	out = open(argvs[2],"w")

	pool = mp.Pool(proc)
	e = partial(write, documents,proc)
	callback = pool.map(e, range(proc))
	
	for i in callback:
		for j in i:
			out.write(j)

main(int(argvs[3]))

