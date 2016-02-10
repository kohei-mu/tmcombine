#/usr/bin/env python
#-*- coding: utf-8 -*-

""" This program adds topic based feature to n-best list using document topic distribution. The topic distribution is extracted by learning LDA with pivot side training corpus of bilingual corpus and assigning topic distribution of the training document to phrase-pair, which appears in the training document. I implemented four different calculation modes(bunsan, sub_cos, average_cos, min_cos). Mode bunsan calculates the variance of cosine distance of each topic distribution. Mode sub_cos calculates average of cosine distance, and adds max of between each cosine distance and the average cosine distance as the feature. Mode average_dis calculates average of topic distribution, and adds min of between the average and each distribution as the feature. Mode min_cos calculates the cosine distance of topic distribution, and adds min of them as the feature. """

import numpy as np
import sys, re, itertools, argparse, scipy.spatial.distance
reload(sys)
sys.setdefaultencoding("utf-8")
from functools import partial
import multiprocessing as mp
np.seterr(divide='ignore', invalid='ignore')
from math import log

parser = argparse.ArgumentParser(description="show this help message")
parser.add_argument("-s", "--source", dest="src",help="source sentences")
parser.add_argument("-t", "--table", dest="pt",help="merged phrase table with distribution")
parser.add_argument("-tpnum", dest="tpnum", type=int, help="number of topic model")
parser.add_argument("-nb", "--nbest", dest="nbest", help="n-best list")
parser.add_argument("-p", "--proc", dest="proc", type=int, help="processes for multiprocessing of making disDic")
parser.add_argument("-nbo", "--nbest_out", dest="nbest_out", help="output n-best list with new feature")
parser.add_argument("-mode", dest="mode",choices=["bunsan", "sub_cos", "average_cos", "min_cos"], default="bunsan", help="how to compute topic feature")
args = parser.parse_args()

p = re.compile(r'\|.*?\|')
number = re.compile(r'\d+')
f_deli = re.compile(r'\w+:')

def n_best(line):
	p = re.compile(r'\|.*?\|')
	sentence = line.split(" ||| ")
	phrases = p.split(sentence[1].strip())

	return phrases

def main():
	n_best_file = open(args.nbest, "r")
	source_file = open(args.src,"r").readlines()
	n_best_out = open(args.nbest_out, "w")

	dic = make_dic()

	for line in n_best_file:
		dis = 0
		target_list = n_best(line)
		source_num = int(line.split("|||")[0].rstrip())
		source = source_file[source_num].split()
		features = [f.strip() for f in f_deli.split(line.split("|||")[2].strip())]
		features = " ".join(features)
		translation = line.split("|||")[1]
		aligns = p.findall(translation)

		dis_list = []
		target_sen = str()

		for align, target in zip(aligns, target_list):
			target_sen += target.strip() + " "
			align_num = number.findall(align)
			start = int(align_num[0])
			end = int(align_num[1])+1
			source_words = source[start:end]
			key = " ".join(source_words), "".join(target.strip())
			try:
				dis = np.array(dic[key],dtype=np.float)
			except:
				pass
			if not (dis == np.array([0]*args.tpnum, dtype=np.float)).all():
				dis_list.append(dis)

		if args.mode == "bunsan":
			score = bunsan(dis_list)
		elif args.mode == "sub_cos":
			score = sub_cos(dis_list)
		elif  args.mode == "average_cos":
			score = average_cos(dis_list)
		elif args.mode == "min_cos":
			score = min_cos(dis_list)
		
		n_best_out.write(outline)


def bunsan(dis_list):
	cos_list = []
	pair = list(itertools.combinations(dis_list, 2))
	for i in pair:
		cos_dis = scipy.spatial.distance.cosine(i[0], i[1])
		cos_list.append(cos_dis)

	if len(cos_list) > 1:
		cos_av = sum(cos_list) / len(cos_list)
		bunsan = sum([abs(cos_av - cos) for cos in cos_list]) / len(cos_list)
		try:
			score = log(bunsan)
		except:
			score = float(0.0)
	else:
		score = float(0.0)

	if np.isnan(score):
		score = float(0.0)

	return score

def sub_cos(dis_list):
	cos_list = []
	pair = list(itertools.combinations(dis_list, 2))
	for i in pair:
		cos_dis = scipy.spatial.distance.cosine(i[0], i[1])
		cos_list.append(cos_dis)

	if len(cos_list) > 1:
		cos_av = sum(cos_list) / len(cos_list)
		sub = 0
		for cos in cos_list:
			tmp_sub = abs(cos_av - cos)
			if tmp_sub > sub: sub = tmp_sub

		try:
			score = log(sub)
		except:
			score = float(0.0)
	else:
		score = float(0.0)

	if np.isnan(score):
		score = float(0.0)

	return score

def average_cos(dis_list):
	if len(dis_list) > 1:
		dis_av = sum(dis_list) / len(dis_list)
		cos = 1 
		for dis in dis_list:
			tmp_cos = 1 - scipy.spatial.distance.cosine(dis_av, dis)
			if tmp_cos < cos: cos = tmp_cos

		try:
			score = log(cos)
		except:
			score = float(0.0)
	else:
		score = float(0.0)

	if np.isnan(score):
		score = float(0.0)

	return score

def min_cos(dis_list):
        cos_list = []
        pair = list(itertools.combinations(dis_list, 2))
        for i in pair:
                cos_dis = 1 - scipy.spatial.distance.cosine(i[0], i[1])
                cos_list.append(cos_dis)

        if len(cos_list) > 1:

                try:
                        score = log(min(cos_list))
                except:
                        score = float(0.0)
        else:
                try:
                        score = log(cos_dis)
                except:
                        score = float(0.0)

        if np.isnan(score):
                score = float(0.0)

        return score

def make_keyList():

	key_list = []
	tmp_n_best = open(args.nbest,"r")
	tmp_source_file = open(args.src,"r").readlines()
	sys.stderr.write("Start making key-list ...\n")
	for line in tmp_n_best:
		target_list = n_best(line)
		source_num = int(line.split("|||")[0].rstrip())
		source = tmp_source_file[source_num].split()
		translation = line.split("|||")[1]
		aligns = p.findall(translation)

		dis_list = []
		cos_list = []

		for align, target in zip(aligns, target_list):
			align_num = number.findall(align)
			start = int(align_num[0])
			end = int(align_num[1])+1
			source_words = source[start:end]

			key = " ".join(source_words), "".join(target.strip())
			key_list.append(key)
	sys.stderr.write("Done!\n")

	return list(set(key_list))

def make_dic():
	key_list = make_keyList()
	dc = {}
	sys.stderr.write("Start making dic...\n")
	for i in open(args.pt,"r"):
		i = i.split(" ||| ")
		source = i[0].strip()
		target = i[1].strip()
		key = source, target
		if key in key_list:
			dis = np.array(i[-1].strip().split(), dtype=np.float)
			dc[key] = dis
			key_list.remove(key)
	sys.stderr.write("Done!\n")

	return dc



if __name__ == "__main__":
	main()
	

