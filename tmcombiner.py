#/usr/bin/env python
#-*- coding: utf-8 -*-

# This program combines two phrase-tables
# Mode
# 0 -> normal combination(baseline) combine tables (remove null alignments) -> merge multiple rules -> normalize probabilities -> calculate new lexical translation probabilities using Wu(2007)'s phrase method
# 1 -> pvt word topic distribution for word disambiguation. using n-best-score.py, add topic based new features to n-best-list, and rerank.
# 2,3,4 -> pvt document topic distribution for word disambiguation. using n-best-score.py, add topic based new features to n-best-list, and rerank. 2 for pvt-tgt side topic distribution, 3 for pvt-src side topic distribution and 4 for both pvt-src/tgt side topic distribution.
# 5 -> add max probabilities of pvt-src/tgt side topic distribution to phrase-table as new features.
# 6 -> Huang et al.(2013)'s phrase probability induction method using topic model.

import sys, os, gzip, copy, re, argparse, itertools, scipy.spatial.distance
from math import log, exp, sqrt
from collections import defaultdict, Counter
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from functools import partial
import multiprocessing as mp

def parse_command_line():
	parser = argparse.ArgumentParser(description="help messages")
	group1 = parser.add_argument_group('Main options')
	group2 = parser.add_argument_group('More model combination options')

	group1.add_argument("-ps", dest="pvt_src", help="pivot-source phrase-table")
	group1.add_argument("-pt", dest="pvt_tgt", help="pivot-target phrase-table")
	group1.add_argument("-o", dest="output", default="tmp_phrase-table", type=str, help="tmp output phrase-table(DEFAULT:%(default)s)")
	group1.add_argument("-m", dest="merged_output", default="merged_phrase-table", type=str, help="mergerd output phrase-table(DEFAULT:%(default)s)")
	group1.add_argument("-n", dest="norm_output", default="normalized_phrase-table", type=str, help="probabilitiy normalized output phrase-table(DEFAULT:%(default)s)")
	group1.add_argument("-l", dest="lex_output", default="final_phrase-table", type=str, help="lexical probability calculated phrase-table(DEFAULT:%(default)s)")
	group1.add_argument("-invert", dest="invert_flag", choices=["yes","no"],default="yes", help="invert model or not(DEFAULT:%(default)s)")
	group1.add_argument("-combine_method", dest="combine", choices=["sum","max"], default="sum", help="combine method to combine tables(DEFAULT:%(default)s)")
	group1.add_argument("-tp", dest="tp", choices=[0, 1, 2, 3, 4, 5, 6], default=0, type=int, help="0->not use topic model(baseline); 1->use word topic distribution; 2->use pvt-tgt topic distribution; 3->use src-pvt topic distribution; 4->use both pvt-src/tgt topic distribution; 5->use max topic probabilities as features(both src/tgt side); 6->use huang's phrase probability induction method")

	group2.add_argument("-disfile", dest="disfile", type=str, help="word distribution file(for only mode tp 1)")
	group2.add_argument("-tpnum", dest="tpnum", default=20, type=int, help="topic number(for mode tp 1, 2, 3, 4, 5, 6)")
	group2.add_argument("-p", dest="proc", default=10, type=int, help="process number for multiprocessing")
	group2.add_argument("-s_doc", dest="src", type=str, help="src training document(for mode tp 3, 4, 5, 6)")
	group2.add_argument("-p_doc_s", dest="s_pvt", type=str, help="pivot(src) training document(for mode tp 3, 4, 5, 6)")
	group2.add_argument("-s_theta", dest="s_theta", type=str, help="pvt theta file of pvt-src side(documents topic distribution, for mode tp 3, 4, 5, 6)")
	group2.add_argument("-t_doc", dest="tgt", type=str, help="tgt training document(for mode tp 2, 4, 5, 6)")
	group2.add_argument("-p_doc_t", dest="t_pvt", type=str, help="pivot(tgt) training document(for mode tp 2, 4, 5, 6)")
	group2.add_argument("-t_theta", dest="t_theta", type=str, help="pvt theta file of pvt-tgt side(documents topic distribution, for mode tp 2, 4, 5, 6)")
	return parser.parse_args()


####### global functions ###############################################################

def handle_file(filename, action, mode='r'):
	""" open/close files and select read/write mode """

	if action == 'open':
		if mode == 'r':
			mode = 'rb+'
		elif mode == 'w':
			mode = 'wb+'

		if mode == 'rb+' and not os.path.exists(filename):
			if os.path.exists(filename + '.gz'):
				filename = filename + '.gz'
			else:
				sys.stderr.write("Error: unable to open file. " + filename + " - aborting.z\n")
				exit(1)

		if filename.endswith('.gz'):
			fileobj = gzip.open(filename, mode)
		else:
			fileobj = open(filename, mode)

		return fileobj

	elif action == 'close':
		filename.close()

def _load_line(line):
	""" take one rule line and make it into the readable format as a list """

	if not line:
		return None

	line = line.strip().split('|||')

	if line[-1].endswith(" |||"):
		line[-1] = line[-1][:-4]
		line.append("")

	#src
	line[0] = line[0].strip()
	#tgt
	line[1] = line[1].strip()

	#features
	line[2] = [float(i) for i in line[2].strip().split(" ")]

	#alignment
	phrase_align = defaultdict(lambda: []*3)
	for pair in line[3].strip().split(" "):
		try:
			s,t = pair.split("-")
			s,t = int(s),int(t)
			phrase_align[s].append(t)
		except:
			pass
	line[3] = phrase_align

	#word counts
	line[4] = [long(float(i)) for i in line[4].strip().split(" ")]

	if len(line) == 7:
		line[5] = line[5].strip() # pivot phrase
		line[6] = [float(i) for i in line[6].strip().split(" ")] # document/word topic distribution
	elif len(line) == 6:
		line[5] = [float(i) for i in line[5].strip().split(" ")] # document topic distribution

	return line


def _write_phrasetable_file(line):
	""" take a list as an input and write it into phrase-table format """

	if args.tp == 1 or args.tp == 2 or args.tp == 3 or args.tp == 4:
		src, tgt, features, alignment, word_counts, pvt, tp = line[:7]

	elif args.tp == 0 or args.tp == 6 or args.tp == 5:
		src, tgt, features, alignment, word_counts = line[:5]

	features = " ".join(["%.6g" %(f) for f in features])

	alignments = []
	for src_id, tgt_id_lst in alignment.iteritems():
		for tgt_id in sorted(tgt_id_lst):
			alignments.append(str(src_id) + "-" + str(tgt_id))
	
	extra_space = ""
	if len(alignments) > 1:
		extra_space = ""
	alignments = " ".join(str(x) for x in alignments)

	word_counts = " ".join(["%.6g" %(f) for f in word_counts])

	if args.tp == 1 or args.tp == 2 or args.tp == 3 or args.tp == 4:
		tp  = " ".join(["%.6g" %(f) for f in tp])
		outline = "%s ||| %s ||| %s ||| %s%s ||| %s ||| %s ||| %s\n" %(src, tgt, features, alignments, extra_space, word_counts, pvt, tp)

	elif args.tp == 0 or args.tp == 6 or args.tp == 5:
		outline = "%s ||| %s ||| %s ||| %s%s ||| %s\n" %(src, tgt, features, alignments, extra_space, word_counts)

	return outline

def table_sort(model, w_model,flag=None, key=None):
	""" sort phrase-table and invert src/tgt order """

	dic = {}
    	dic_count = 0
    	write_model = handle_file(w_model, "open", "w")

    	for i in model:
			if flag == "ps_invert":
				i = inverse_model(i)
				i = i.strip().split("|||")
			else:
				i = i.strip().split("|||")
		
			if key == "tgt":
				dic[i[1], i[0], dic_count] = i
			else:
				dic[i[0], i[1], dic_count] = i
			dic_count += 1

    	num = len(dic)
    	count = 1

    	#dicをabc降順、フレーズの長さの長い順に並べ替え
    	for j in sorted(dic.keys(), reverse=True, key=lambda x :(x[0],x[1])):
		if len(dic[j]) == 7:
			src, tgt , features, alignments, word_counts, pvt, tp = dic[j][:7]
			line = "%s|||%s|||%s|||%s|||%s|||%s|||%s" %(src,tgt,features,alignments,word_counts, pvt, tp)
		elif len(dic[j]) == 6:
			src, tgt , features, alignments, word_counts, tp = dic[j][:6]
			line = "%s|||%s|||%s|||%s|||%s|||%s" %(src,tgt,features,alignments,word_counts, tp)
		elif len(dic[j]) == 5:
    			src, tgt , features, alignments, word_counts = dic[j][:5]
    			line = "%s|||%s|||%s|||%s|||%s" %(src,tgt,features,alignments,word_counts)                

       		if count < num:line = line + "\n"
        	count += 1
        	write_model.write(line)

    	handle_file(write_model, "close",mode="w")

def inverse_model(line):
	""" invert src/tgt order of phrase-table  """

	line = _load_line(line)
	
	#src/tgt
	pvt_word = line[1].strip()
	line[1] = line[0].strip()
	line[0] = pvt_word

    	#features
	features = line[2]
	tmp = features[0]
	features[0] = features[2]
	features[2] = tmp
	tmp = features[1]
	features[1] = features[3]
	features[3] = tmp

    	#alignment
	phrase_align = defaultdict(lambda: []*3)
	for src_word, tgt_list in line[3].iteritems():
		for tgt_word in tgt_list:
			phrase_align[tgt_word].append(src_word)

    	#sometimes, the count is too big
	if (len(line[4]) > 1):
		tmp = line[4][0]
		line[4][0] = line[4][1]
		line[4][1] = tmp
	
	#write into the table format
	features = " ".join(["%.6g" %(f) for f in features])
	alignments = []
	for src_id, tgt_id_lst in phrase_align.iteritems():
		for tgt_id in sorted(tgt_id_lst):
			alignments.append(str(src_id) + "-" + str(tgt_id))

	extra_space = ""
	if len(alignments) > 1:extra_space = ""
	alignments = " ".join(str(x) for x in alignments)
	word_counts = " ".join(["%.6g" %(f) for f in line[4]])	
	
	if args.tp == 3 or args.tp == 4 or args.tp == 6 or args.tp == 5:
		tp  = " ".join(["%.6g" %(f) for f in line[5]])
		outline = "%s|||%s|||%s|||%s%s|||%s|||%s\n" %(line[0], line[1], features, alignments,extra_space, word_counts,tp)
	else:
		outline = "%s|||%s|||%s|||%s%s|||%s\n" %(line[0], line[1], features, alignments,extra_space, word_counts)
	return outline

def makeDic_p(model, proc, p):
	""" make pvt:word dis dictionary by multiprocessing  """

	modelLength = len(model)
	ini = modelLength * p / proc
	fin = modelLength * (p+1) /proc
	tmp_dic = {}

	for i in model[ini:fin+1]:
		i = i.split("\t")
		pvt = i[0].strip()
		tmp_dic[pvt] = np.array(i[1:], dtype=np.float)

	return tmp_dic

def disToDic(proc):
	""" make word distribution dictionary using makeDic_p function(need for mode 1)  """

	model = open(args.disfile, "r").readlines()
	pool = mp.Pool(proc)
	e = partial(makeDic_p, model,proc)
	callback = pool.map(e, range(proc))

	dic = {}
	for i in callback:
		dic.update(i)

	return dic
	
def lda_process(src_tgt, pvt, dis,  model, proc,flag, p):
	""" add document topic distribution to phrase-table """

	length = len(model)
	ini = length * p / proc
	fin = length * (p+1) / proc
	lst = []

	for i in model[ini:fin+1]:
		line = i.strip().split(" ||| ")

		if args.tp == 3:
			srctgt = line[0].strip()
                	pivot = line[1].strip()

       		elif args.tp == 2:
                	pivot = line[0].strip()
                	srctgt = line[1].strip()

		elif (args.tp == 4 or args.tp == 5 or args.tp == 6)  and flag == "ps":
			srctgt = line[0].strip()
                        pivot = line[1].strip()

		elif (args.tp == 4 or args.tp == 5 or args.tp == 6)  and flag == "pt":
			pivot = line[0].strip()
                        srctgt = line[1].strip()

        	tp_count = 0
        	tmp_dis = np.array([0]*args.tpnum,dtype=np.float)
        	for f in range(len(dis)):
                	if srctgt in src_tgt[f] and pivot in pvt[f]:
                        	topic = np.array(dis[f].strip().split(), dtype=np.float)
                        	tp_count += 1
				c = min(src_tgt[f].count(srctgt), pvt[f].count(pivot)) # count(I,D)
				tmp_dis += c * topic
        	if tp_count > 0:
			#normalize
			tp_sum = sum(tmp_dis)
			tp =  " ".join(["%.6g" %(f) for f in [i / tp_sum  for i in tmp_dis] ])
        	elif tp_count == 0:
			continue

        	line.append(tp)
        	outline = " ||| ".join(line)
		lst.append(outline)
	return lst

def lda_main(src_tgt, pvt, dis,  table, proc, w, flag=None):
	""" add document topic distribution to phrase table using lda_process function(nedd for mode 2,3,4,5,6)  """

	w_file = handle_file(w, "open", mode="w")
        model = table.readlines()
	pool = mp.Pool(proc)
        e = partial(lda_process, src_tgt, pvt, dis, model,proc, flag)
        callback = pool.map(e, range(proc))

	for i in callback:
		for j in i:
			w_file.write(j+"\n")
	
########################################################################

class Triangulate_TMs():
	""" This class takes two phrase-tables as inputs and combines them into one phrase-table. 
	    Null alignments are removed. In the case of mode 1~6, topic calculation is held at first. """

	def __init__(self, model1=None, 
			   model2=None, 
             		   output_file=None,
			   invert_flag=None, 
			   disDic=None, 
			   tpnum=None, 
			   src=None, 
			   tgt=None, 
			   s_pvt=None,
			   t_pvt=None, 
			   s_theta=None, 
			   t_theta=None):

		self.output_file = output_file
		self.model1 = model1
		self.model2 = model2
		self.invert_flag = invert_flag
		self.disDic = disDic
		self.tpnum = tpnum

		if args.tp == 2:
			self.tgt = handle_file(tgt, "open", "r").readlines()
			self.t_pvt = handle_file(t_pvt, "open", "r").readlines()
			self.t_theta = handle_file(t_theta, "open", "r").readlines()
		elif args.tp == 3:
			self.src = handle_file(src, "open", "r").readlines()
			self.s_pvt = handle_file(s_pvt, "open", "r").readlines()
			self.s_theta = handle_file(s_theta, "open", "r").readlines()
		elif args.tp == 4 or args.tp == 6 or args.tp == 5:
			self.tgt = handle_file(tgt, "open", "r").readlines()
			self.t_pvt = handle_file(t_pvt, "open", "r").readlines()
			self.t_theta = handle_file(t_theta, "open", "r").readlines()
			self.src = handle_file(src, "open", "r").readlines()  
                        self.s_pvt = handle_file(s_pvt, "open", "r").readlines()
			self.s_theta = handle_file(s_theta, "open", "r").readlines()

	def combine_standard(self):
		model1 = handle_file(self.model1, 'open', 'r')
		model2 = handle_file(self.model2, 'open', 'r')

		if self.invert_flag == "yes":
			if args.tp == 4 or args.tp == 6 or args.tp == 5:
                                lda_main(self.src, self.s_pvt, self.s_theta, model1, args.proc, "model1_dis", flag="ps")
				model1_dis = handle_file("model1_dis", "open", "r")
				table_sort(model1_dis, "model1_dis_inv",flag="ps_invert")
                                model1 = handle_file("model1_dis_inv", "open", "r")
                                lda_main(self.tgt, self.t_pvt, self.t_theta, model2, args.proc, "model2_dis", flag="pt")
                                model2_dis = handle_file("model2_dis", "open", "r")
				table_sort(model2_dis, "model2_dis_sorted")                                           
                                model2 = handle_file("model2_dis_sorted", "open", "r")  
			else:
				if args.tp == 3:
					lda_main(self.src, self.s_pvt, self.s_theta, model1, args.proc, "model1_dis")
					model1_dis = handle_file("model1_dis", "open", "r")
					table_sort(model1_dis, "model1_dis_inv",flag="ps_invert")
					model1 = handle_file("model1_dis_inv", "open", "r")
				else:
					table_sort(model1, "model1_inv",flag="ps_invert")
                        		model1 = handle_file("model1_inv", "open", "r")

				if args.tp == 2:
					lda_main(self.tgt, self.t_pvt, self.t_theta, model2, args.proc, "model2_dis")
					model2_dis = handle_file("model2_dis", "open", "r")
					table_sort(model2_dis, "model2_dis_sorted")
					model2 = handle_file("model2_dis_sorted", "open", "r")
				else:
					table_sort(model2, "model2_sorted")
                        		model2 = handle_file("model2_sorted", "open", "r")

		output_file = handle_file(self.output_file, 'open', mode='w')

		self._write_phrasetablefile(model1, model2, output_file)
		handle_file(output_file, 'close', mode='w')

		return output_file

	def _write_phrasetablefile(self, model1, model2, output_object):
		
		self.phrase_equal = defaultdict(lambda: []*3)
		self._phrasetable_traversal(model1, model2, output_object)
		sys.stderr.write("Done\n")

	def _phrasetable_traversal(self, model1, model2, output_object):
		""" combine two phrase-tables  """

		#src
		line1 = _load_line(model1.readline())
		#tgt
		line2 = _load_line(model2.readline())
		count = 0

		while(1):
			if not count % 100000:
				sys.stderr.write(str(count) + "...")
			count += 1

			if self.phrase_equal[0]:
				if line1 and line1[0] == self.phrase_equal[0]:
					self.phrase_equal[1].append(line1)
					line1 = _load_line(model1.readline())
					continue
				elif line2 and line2[0] == self.phrase_equal[0]:
					self.phrase_equal[2].append(line2)
					line2 = _load_line(model2.readline())
					continue
				else:
					self._combine_and_print(output_object)				

			if not line1 or not line2:
				self._combine_and_print(output_object)
				sys.stderr.write("Finish loading\n")
				return None

			if not self.phrase_equal[0]:
				if line1[0] == line2[0]:
					self.phrase_equal[0] = line1[0]
				elif line1[0].startswith(line2[0]):
					line1 = _load_line(model1.readline())
				elif line2[0].startswith(line1[0]):
					line2 = _load_line(model2.readline())
				elif line2[0] < line1[0]:
					line1 = _load_line(model1.readline())
				elif line1[0] < line2[0]:
					line2 = _load_line(model2.readline())

	def _combine_and_print(self, output_object):
		""" combine two lines that have same pivot phrase and write into phrase-table format  """		

		for phrase1 in self.phrase_equal[1]:
			for phrase2 in self.phrase_equal[2]:
				if (phrase1[0] != phrase2[0]):
					sys.exit("The pivots are different")

				pvt = self.phrase_equal[0]
				src, tgt = phrase1[1], phrase2[1]
				features = self._get_features_Cohn(phrase1[2],phrase2[2])
				word_alignments = self._get_word_alignments(phrase1[3],phrase2[3])
				
				# phrase pairs containing mismatched alignments(all indexes of the src phrase align to NULL) are deleted
				align_src = len(word_alignments.keys())
				align_tgt = len(set([ j for i in word_alignments.values() for j in i]))
				if align_src == 0 or align_tgt == 0: continue
	
				word_counts = self._get_word_counts(phrase1[4], phrase2[4])

				if args.tp == 1:
					word_dis = self.calc_dis("".join(pvt))
					outline = _write_phrasetable_file([src, tgt, features, word_alignments, word_counts, pvt, word_dis])

				elif args.tp == 0:
					outline = _write_phrasetable_file([src, tgt, features, word_alignments, word_counts])

				elif args.tp == 2:
					doc_dis = phrase2[5]
					outline = _write_phrasetable_file([src, tgt, features, word_alignments, word_counts, pvt, doc_dis])

				elif args.tp == 3:
					doc_dis = phrase1[5]
                                        outline = _write_phrasetable_file([src, tgt, features, word_alignments, word_counts, pvt, doc_dis])

				elif args.tp == 4:
					doc_dis = self.topic_plus(phrase1[5], phrase2[5])
					outline = _write_phrasetable_file([src, tgt, features, word_alignments, word_counts, pvt, doc_dis])

				elif args.tp == 5:

					tp_s = max(phrase1[5])	
					features.append(tp_s)				

					tp_t = max(phrase2[5])
					features.append(tp_t)

					outline = _write_phrasetable_file([src, tgt, features, word_alignments, word_counts])
				elif args.tp == 6:
					features = self.huang(phrase1[2], phrase2[2], phrase1[5], phrase2[5])
					outline = _write_phrasetable_file([src, tgt, features, word_alignments, word_counts])
		
				output_object.write(outline)

		self.phrase_equal = None
		self.phrase_equal = defaultdict(lambda: []*3)

	def huang(self, features1, features2, tp1, tp2):
		""" use similarity of document topic distribution for phrase translation probability induction(Huang et al.(2013)) (for mode 6) """

		phrase_features = [0]*5

		#lexical translation probabilities(recomputed later by Phrase method)
		phrase_features[1] = features1[3] * features2[1]
		phrase_features[3] = features1[1] * features2[3]

		#phrase penalty
		phrase_features[4] = features1[4]
		
		#phrase translation probabilities
		phrase_features[0] = 1 - scipy.spatial.distance.cosine(tp1, tp2) # sim(sp,pt)
		phrase_features[2] = 1 - scipy.spatial.distance.cosine(tp1, tp2) # sim(sp,pt)

		return phrase_features

	def topic_plus(self, ps, pt):
		""" pvt-src side topic distribution plus pvt-tgt side topic distribution (for mode 4) """
	
		ps_pt = np.array(ps, dtype=np.float) + np.array(pt, dtype=np.float)
		sum_tp = sum(ps_pt)
		norm_tp = [i / sum_tp  for i in ps_pt]
		return norm_tp

	def calc_dis(self, pivot):
		""" return word topic distribution (for mode 1) """

		words = pivot.split()

		#phrase
		if len(words) > 1:
			try:
				dis = self.heikin(words)

			except:
				dis = np.array([0]*args.tpnum, dtype=np.float)
		#word
		else:
			try:
				dis = np.array(self.disDic["".join(words)], dtype=np.float)
			except:
				dis = np.array([0]*args.tpnum, dtype=np.float)

		return dis

	def heikin(self, words):
		""" calculate average of word topic distribution(for mode 1) """	

		dis = 0
		count = 0
		for i in words:
			try:
				dis += np.array(self.disDic["".join(i)], dtype=np.float)
				count += 1
			except:
				pass

		if count != 0:
			phrase_dis = np.array((dis / count), dtype=np.float)
		else:
			phrase_dis = np.array([0]*args.tpnum, dtype=np.float)

		return phrase_dis

	def _get_features_Cohn(self, feature1, feature2):
		""" calculate features in a conventional way """		

		phrase_features = [0]*5

   		phrase_features[0] = feature1[2] * feature2[0] 	# backward phrase translation probability
		phrase_features[1] = feature1[3] * feature2[1] 	# backward lexical translation probability
   		phrase_features[2] = feature1[0] * feature2[2]	# forward phrase translation probability
  		phrase_features[3] = feature1[1] * feature2[3]	# forward lexical translation probability
		phrase_features[4] = feature1[4] # (feature2[4]) 	# phrase penalty

   		return phrase_features

   	def _get_word_alignments(self, phrase_ps, phrase_pt):
		""" phrase pairs containing null alignments are automatically removed in _combine_and_print function """		
		phrase_st = defaultdict(lambda: []*3)
		for pvt_id, src_lst in phrase_ps.iteritems():
			if pvt_id in phrase_pt:
				tgt_lst = phrase_pt[pvt_id]
				for src_id in src_lst:
					for tgt_id in tgt_lst:
						if (tgt_id not in phrase_st[src_id]):
							phrase_st[src_id].append(tgt_id)


		return phrase_st

	def _get_word_counts(self, count1, count2):

		word_count = [0]*2

		word_count[0] = count2[0]
		word_count[1] = count1[0]

		return word_count

##########################################################################################

class Merge_TM():
	""" This class takes combined phrase-table as an input and merges multiple lines(rules) into one line.
	 The output is one cleaned phrase-table. """

	def __init__(self, model=None, output_file=None):

		self.model = model
		self.output_file = output_file

	def _combine_TM(self, flag=False, prev_line=None):

		prev_line = []
		sys.stderr.write("\nCombine Multiple lines by combine_given_weights " + "\n")
		output_object = handle_file(self.output_file, "open", mode="w")
		sys.stderr.write("Start mergin multiple lines ...")
		
		if args.tp == 1 or args.tp == 2 or args.tp == 3 or args.tp == 4:
			self._regular_traversal_pivot(flag, prev_line, output_object)

		elif args.tp == 0 or args.tp == 5 or args.tp == 6:
			self._regular_traversal(flag, prev_line, output_object)
		
		handle_file(output_object, "close", mode="w")

	def _regular_traversal(self, flag=None, prev_line=None, output_object=None):
		""" merge multiple lines into one line (for mode 0, 5, 6) """

		count = 0
		for line in self.model:
			if not count % 100000:
				sys.stderr.write(str(count) + "...")
			count += 1

			line = _load_line(line)
			if flag:
				if line[0] == prev_line[0] and line[1] == prev_line[1]:

					if args.combine == "max":
						pre1v_line = self._combine_max(prev_line, line) # select max feature
					else:
						#default
						prev_line = self._combine_sum(prev_line, line) # sum up features
					continue
				else:
					outline = _write_phrasetable_file(prev_line)
					output_object.write(outline)
					prev_line = line
					flag = False
			elif prev_line:
				if line[0] == prev_line[0] and line[1] == prev_line[1]:

					if args.combine == "max":
						prev_line = self._combine_max(prev_line, line) # select max feature
					else:
						#default
						prev_line = self._combine_sum(prev_line, line) # sum up features
					flag = True
					continue
				else:
					outline = _write_phrasetable_file(prev_line)
					output_object.write(outline)
					prev_line = line
			else:
				prev_line = line

		if prev_line:
			outline = _write_phrasetable_file(prev_line)
			output_object.write(outline)
		sys.stderr.write("Done\n")

		return None

	def _regular_traversal_pivot(self, flag=None, prev_line=None, output_object=None):
		""" merge multiple lines into one line (for mode 1, 2, 3, 4) """

		count = 0
		same_list = []
		for line in self.model:
			if not count % 100000:
				sys.stderr.write(str(count) + "...")
			count += 1
			line = _load_line(line)
			if flag:
				if line[0] == prev_line[0] and line[1] == prev_line[1]:
					same_list.append(prev_line)
					prev_line = line
					continue
				else:
					same_list.append(prev_line)
					prev_line = self.combine_same_pivot(same_list) # combine lines that have same phrase pairs by two method
					if not (prev_line[-1] == np.array([0]*args.tpnum, dtype=np.float)).all():
						outline = _write_phrasetable_file(prev_line)
						output_object.write(outline)
					prev_line = line
					flag = False
					same_list = []
			elif prev_line:
				if line[0] == prev_line[0] and line[1] == prev_line[1]:
					same_list.append(prev_line)
					prev_line = line
					flag = True
					continue
				else:
					if not (prev_line[-1] == np.array([0]*args.tpnum, dtype=np.float)).all():
						outline = _write_phrasetable_file(prev_line)
						output_object.write(outline)
					prev_line = line
			else:
				prev_line = line

		if prev_line:
			if flag == True:
				same_list.append(prev_line)
				prev_line = self.combine_same_pivot(same_list) # combine lines that have same phrase pairs by two method
				if not (prev_line[-1] == np.array([0]*args.tpnum, dtype=np.float)).all():
					outline = _write_phrasetable_file(prev_line)
					output_object.write(outline)
			else:
				if not (prev_line[-1] == np.array([0]*args.tpnum, dtype=np.float)).all():
					outline = _write_phrasetable_file(prev_line)
					output_object.write(outline)
		sys.stderr.write("Done\n")

		return None

	def combine_same_pivot(self, lines):
		""" combine lines that have same phrase pairs by two method (for mode 1,2,3,4) """

		src, tgt = lines[0][0:2]
		count = lines[0][4]
		pvt = zip(*lines)[5]

		sum_dis = 0
		sum_ph = 0
		for i in lines:
			dis = np.array(i[-1], dtype=np.float)
			if not (dis == np.array([0]*args.tpnum, dtype=np.float)).all():
				sum_ph += i[2][2]
				sum_dis += dis * i[2][2] # topic distribution * forward phrase translation probability

		if type(sum_dis) != int() and sum_ph != 0:
			sum_dis_norm = sum( np.array((sum_dis / sum_ph), dtype=np.float) ) # for normalization
			dis_av =  [ i / sum_dis_norm for i in np.array((sum_dis / sum_ph), dtype=np.float) ]
		else:
			dis_av = np.array([0]*args.tpnum, dtype=np.float)

		#features
		cur_line = lines[0]
		if args.combine == "max": # select max feature
			for j in range(len(lines)-1):
                        	for i in range(4):
                                	cur_line[2][i] = max(cur_line[2][i], lines[j+1][2][i])

		elif args.combine == "sum": # sum up features (default)
			for j in range(len(lines)-1):
                        	for i in range(4):
                                	cur_line[2][i] += lines[j+1][2][i]
                                	cur_line[2][i] = min(cur_line[2][i], 1.0)

		#alignment
		for i in range(len(lines)-1):
                	for _src, key in lines[i+1][3].iteritems():
                        	for _tgt in key:
                                	if _tgt not in cur_line[3][_src]:
                                        	cur_line[3][_src].append(_tgt)

		outline = [src, tgt, cur_line[2], cur_line[3], count, pvt, dis_av]		
		return outline

	def _combine_sum(self, prev_line=None, cur_line=None):
		""" sum up features """

		for i in range(4):
			prev_line[2][i] += cur_line[2][i]
			prev_line[2][i] = min(prev_line[2][i], 1.0)
		if args.tp == 5:
			for i in range(5, 7): # i[2][5], i[2][6]
				prev_line[2][i] += cur_line[2][i]
				prev_line[2][i] = min(prev_line[2][i], 1.0)
		
		for src, key in cur_line[3].iteritems():
			for tgt in key:
				if tgt not in prev_line[3][src]:
					prev_line[3][src].append(tgt)

		return prev_line

	def _combine_max(self, prev_line=None, cur_line=None):
		""" select max feature """

		for i in range(4):
			prev_line[2][i] = max(prev_line[2][i], cur_line[2][i])
		if args.tp == 5:
			for i in range(5, 7):# i[2][5], i[2][6]
				prev_line[2][i] = max(prev_line[2][i], cur_line[2][i])


		for src, key in cur_line[3].iteritems():
			for tgt in key:
				if tgt not in prev_line[3][src]:
					prev_line[3][src].append(tgt)
		
		return prev_line

####################################################################

class Normalize_prob():
	""" This class takes merged table as input and normalizes phrase translation probabilities. """

	def __init__(self, model=None, output_file=None):
		self.model = model
		self.output_file = output_file

	def normalize_table(self):

		sys.stderr.write("\nNormalize probabilities " + "\n")
		output_object = handle_file(self.output_file, "open", mode="w")
		sys.stderr.write("Start normalizing probabilities ...")
		self.norm_prob(output_object)
		handle_file(output_object, "close", mode="w")
	
	def norm_prob(self, output_object=None):
		""" sort and normalize phrase-table """

		table_sort(self.model, "tmp_norm_tgt",flag=None, key="tgt")
		tmp_model_tgt = handle_file("tmp_norm_tgt", "open", "r")
		tmp_norm_tgt_sorted = handle_file("tmp_norm_tgt_sorted", "open", mode="w")
		
		prev_line = []
		for i in tmp_model_tgt:
			line = _load_line(i)
			try:
				if line[1] == prev_line[0][1]:
					prev_line.append(line)
				elif line[1] != prev_line[0][1]:
					self.calc_norm(prev_line, tmp_norm_tgt_sorted, flag="tgt")
					prev_line = []
					prev_line.append(line)
			except:
				prev_line.append(line)
				
		if prev_line:
			self.calc_norm(prev_line, tmp_norm_tgt_sorted, flag="tgt")
			prev_line = []
		handle_file(tmp_model_tgt, "close", "r")
		os.remove("tmp_norm_tgt")
		handle_file(tmp_norm_tgt_sorted, "close", mode="w")

		tmp_norm_tgt_sorted = handle_file("tmp_norm_tgt_sorted", "open", "r")
		table_sort(tmp_norm_tgt_sorted, "tmp_norm_src", flag=None, key=None)
		os.remove("tmp_norm_tgt_sorted")
		tmp_model_src = handle_file("tmp_norm_src", "open", "r")

		for i in tmp_model_src:
			line = _load_line(i)
			try:
				if line[0] == prev_line[0][0]:
					prev_line.append(line)
				elif line[0] != prev_line[0][0]:
					self.calc_norm(prev_line, output_object, flag="src")
					prev_line = []
					prev_line.append(line)
			except:
				prev_line.append(line)
		if prev_line:
			self.calc_norm(prev_line, output_object, flag="src")
		
		os.remove("tmp_norm_src")
		sys.stderr.write("Done\n")

		return None

	def calc_norm(self, prev_lines, output_object=None, flag=None):
		""" normalize probabilities """

		if flag == "tgt":
			sum_prob = sum([i[2][0] for i in prev_lines])
			for i in prev_lines:
				i[2][0] = float(i[2][0]) / sum_prob
				outline = _write_phrasetable_file(i)
				output_object.write(outline)

		elif flag == "src":
			sum_prob = sum([i[2][2] for i in prev_lines])
			for i in prev_lines:
				i[2][2] = float(i[2][2]) / sum_prob
				outline = _write_phrasetable_file(i)
                                output_object.write(outline)

##########################################################################

class Word_prob():
	""" This class is the final stage of this program, takes one noramlized phrase-table 
	and replaces normally induced lexical translation probabilities with new one using Wu et al.(2007)'s phrase method. """

	def __init__(self, model=None, output=None, model_for_lexDic=None):
		self.model = model
		self.output = output
		self.lex_dic = self.make_lexDic(model_for_lexDic)
	
	def lex_calc(self):
		""" re-computing lexical translation probabilities with Huang et al(2013)'s phrase method """

		sys.stderr.write("\nStart re-computing lexical translation probabilities...\n")
		output_object = handle_file(self.output, "open", "w")
		for i in self.model:
			line = _load_line(i)
			src = line[0].strip().split()
			tgt = line[1].strip().split()

			p_w = 1
			for j in line[3].items():
				tmp_p_w = 0
				for k in j[1]:tmp_p_w += float(self.lex_dic[src[int(j[0])], tgt[int(k)]][0])
				p_w *= (float(tmp_p_w) / len(j[1]) )
			line[2][3] = float(p_w)

			inversed_alignment = defaultdict(lambda: []*3)
			for src_word, tgt_list in line[3].iteritems():
				for tgt_word in tgt_list:
					inversed_alignment[tgt_word].append(src_word)

			p_w = 1
			for j in inversed_alignment.items():	
				tmp_p_w = 0
				for k in j[1]:tmp_p_w += float(self.lex_dic[src[int(k)], tgt[int(j[0])]][1])
				p_w *= (float(tmp_p_w) / len(j[1]) )
			line[2][1] = float(p_w)

			outline = _write_phrasetable_file(line)
			output_object.write(outline)
		sys.stderr.write("Done!\n")

	def make_lexDic(self, model):
		""" make lexical translation probability dictionary(dic[src,tgt]=prob1,prob2) """

        	s2t = handle_file("lex.s2t", "open", "w")
        	t2s = handle_file("lex.t2s", "open", "w")

        	dic = defaultdict(int)
        	for i in model:
                	line = i.strip().split(" ||| ")
                	src=line[0].split()
                	tgt=line[1].split()
                	align = line[3].split()
                	for i in align:
                        	s=int(i.split("-")[0])
                        	t=int(i.split("-")[1])
                        	dic[src[s], tgt[t]] += float(line[2].split()[2])

        	for i, j in sorted(dic.items(), reverse=True, key=lambda x:x[0][0]):
                	outline = "%s ||| %s ||| %s\n" %(i[0], i[1], j)
                	s2t.write(outline)
        	for i, j in sorted(dic.items(), reverse=True, key=lambda x:x[0][1]):
                	outline = "%s ||| %s ||| %s\n" %(i[0], i[1], j)
                	t2s.write(outline)

        	handle_file(s2t, "close", "w")
        	handle_file(t2s, "close", "w")
        	s2t = handle_file("lex.s2t", "open", "r")
        	t2s = handle_file("lex.t2s", "open", "r")
        	s2t_norm = handle_file("lex.s2t.norm", "open", "w")
        	t2s_norm = handle_file("lex.t2s.norm", "open", "w")

        	self.normalize_lex(s2t, s2t_norm, flag="st")
        	self.normalize_lex(t2s, t2s_norm, flag="ts")

        	handle_file(s2t, "close", "r")
        	handle_file(t2s, "close", "r")
        	handle_file(s2t_norm, "close", "w")
        	handle_file(t2s_norm, "close", "w")

        	mergedDic = defaultdict(list)
        	s2t_norm = handle_file("lex.s2t.norm", "open", "r")
        	t2s_norm = handle_file("lex.t2s.norm", "open", "r")

        	for i in s2t_norm:
                	line = i.strip().split(" ||| ")
                	mergedDic[line[0], line[1]].append(float(line[2]))
        	for i in t2s_norm:
                	line = i.strip().split(" ||| ")
                	mergedDic[line[0], line[1]].append(float(line[2]))

        	os.remove("lex.s2t")
        	os.remove("lex.t2s")
        	os.remove("lex.s2t.norm")
        	os.remove("lex.t2s.norm")

        	return mergedDic

	def normalize_lex(self, model, w_file, flag=None):
		""" reorder lexical table in src or tgt order """

        	prev_line=[]
        	for i in model:
                	line = i.strip().split(" ||| ")
               		try:
                        	if flag == "st":
                                	if line[0] == prev_line[0][0]:
                                        	prev_line.append(line)
                                	elif line[0] != prev_line[0][0]:
                                        	self.combine_lex(prev_line, w_file)
                                        	prev_line=[]
                                        	prev_line.append(line)
                        	elif flag == "ts":
                                	if line[1] == prev_line[0][1]:
                                        	prev_line.append(line)
                                	elif line[1] != prev_line[0][1]:
                                        	self.combine_lex(prev_line, w_file)
                                        	prev_line = []
                                        	prev_line.append(line)
                	except:
                        	prev_line.append(line)

        	if prev_line:
                	self.combine_lex(prev_line, w_file)
                	prev_line=[]

	def combine_lex(self,lines, w_lex):
		""" normalize and print lexical table """

        	sum_w = sum([float(i[2])  for i in lines])
        	for i in lines:
                	i[2] = float(i[2]) / sum_w
                	outline = "%s ||| %s ||| %s\n" %(i[0], i[1], i[2])
                	w_lex.write(outline)


###########################  Execute  ###################################

if __name__ == "__main__":

	if len(sys.argv) < 2:
		sys.stderr.write("no command specified. use option -h for usage instructions\n")
	
	else:
		args = parse_command_line()
		
		if args.tp == 1:
			disDic = disToDic(args.proc)
			combiner = Triangulate_TMs(model1=args.pvt_src, 
						   model2=args.pvt_tgt, 
						   output_file=args.output,
						   invert_flag=args.invert_flag, 
						   disDic=disDic,tpnum=args.tpnum)

		elif args.tp == 0:
			combiner = Triangulate_TMs(model1=args.pvt_src, 
						   model2=args.pvt_tgt, 
						   output_file=args.output,
						   invert_flag=args.invert_flag)

		elif args.tp == 2 or args.tp == 3 or args.tp == 4 or args.tp == 5 or args.tp == 6:
			combiner = Triangulate_TMs(model1=args.pvt_src, 
						   model2=args.pvt_tgt, 
					  	   output_file=args.output,
						   invert_flag=args.invert_flag, 
						   tpnum=args.tpnum, 
						   src=args.src, 
						   tgt=args.tgt, 
						   s_pvt=args.s_pvt, 
						   t_pvt=args.t_pvt, 
						   s_theta=args.s_theta, 
						   t_theta=args.t_theta)

		
		#combine tables
		combiner.combine_standard()

		#remove tmp files
                if args.tp == 2:
			pass
                        os.remove("model2_dis")
                        os.remove("model2_dis_sorted")
                        os.remove("model1_inv")
                elif args.tp == 3:
                        os.remove("model1_dis")
                        os.remove("model1_dis_inv")
                        os.remove("model2_sorted")
                elif args.tp == 4 or args.tp == 6 or args.tp == 5:
                        os.remove("model1_dis")
                        os.remove("model1_dis_inv")
                        os.remove("model2_dis")
                        os.remove("model2_dis_sorted")
		elif args.tp == 0:
			os.remove("model1_inv")
			os.remove("model2_sorted")

		#merge multiple lines(rules)
		tmp = handle_file(args.output, "open", "r")
		table_sort(tmp, "tmp_sorted")
		tmp_sorted = handle_file("tmp_sorted", "open", "r")
		merger = Merge_TM(model=tmp_sorted, output_file=args.merged_output)
		merger._combine_TM()

		#normalize probabilities
		#tmp_for_norm = handle_file(args.merged_output, "open", "r")
		#normalizer = Normalize_prob(model=tmp_for_norm, output_file=args.norm_output)
		#normalizer.normalize_table()

		#re-compute lexical translation probabilities
		#tmp_for_lex = handle_file(args.norm_output, "open", "r")
		#model_for_lexDic = handle_file(args.norm_output, "open", "r")
		#lex_calculator = Word_prob(model=tmp_for_lex, output=args.lex_output, model_for_lexDic=model_for_lexDic)
		#lex_calculator.lex_calc()
		
		#delete tmp files
		os.remove("tmp_sorted")
		os.remove(args.output)
		#os.remove(args.merged_output)
		#os.remove(args.norm_output)
