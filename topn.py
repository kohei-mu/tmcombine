#/usr/bin/env python
#-*- coding: utf-8 -*-

# preprocess script for TER

import argparse

parser = argparse.ArgumentParser(description="show this help message")
parser.add_argument("-i", dest="input",help="input file")
parser.add_argument("-w", dest="write", help="write file")
parser.add_argument("-mode", dest="mode", choices=["nbest","cut","ter"], default="cut",help="mode choices")
parser.add_argument("-ter", dest="ter", choices=["1", "0"], default="0", help="delimiter for ter evaluation")
args = parser.parse_args()

f = open(args.input,"r")
w = open(args.write,"w")

def nbestList():
	for i in f:
		line = i.strip().split(" ||| ")
		sentence = line[1]
		if args.ter == "1":
			w.write(sentence+"  "+"( "+str(line[0])+" )"+"\n")
		else:
			w.write(sentence+"\n")

def outputCut():
	count = 0
	for i in f:
		line = i.strip().split("\t")[0]
		if args.ter == "1":
			w.write(line+"  "+"( "+str(count)+" )"+"\n")
		else:
			w.write(line+"\n")
		count += 1

def ter():
	count = 0
	for i in f:
		line = i.strip()
		w.write(line+"  "+"( "+str(count)+" )"+"\n")
		count += 1


if __name__ == "__main__":
	if args.mode == "nbest":
		nbestList()
	elif args.mode == "cut":
		outputCut()
	elif args.mode == "ter":
		ter()


