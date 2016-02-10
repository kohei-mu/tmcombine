#!/bin/bash

""" process doc file into ter format, and evaluate ter score. and evaluate meteor score.  """

#$1 : hyp input
#$2 : ref input 
#$3 : meteor output
#$4 : ter output

#meteor evaluation
java -Xmx2G -jar ~/meteor-1.5/meteor-1.5.jar $1  $2  -l es -lower > $3

#ter evaluation
#topn.py : create ter style format
python $PWD/topn.py -i $1 -w tmp_hyp -mode ter 
python $PWD/topn.py -i $2 -w tmp_ref -mode ter 
java -jar ~/tercom-0.7.2/tercom.7.2.jar -r $PWD/tmp_ref -h $PWD/tmp_hyp > $4
rm tmp_hyp tmp_ref

