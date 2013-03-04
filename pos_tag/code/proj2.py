from nltk.tokenize import word_tokenize, wordpunct_tokenize, sent_tokenize
from sets import Set
import re
import os
import sys
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def getTags(tagfilename = 'alltags'):
    """ return a list of tags

    Read the set of tags from a file that lists all the tags. 
    You could read them from the training file, but this is faster.

    """
    tags = []
    with open(tagfilename,'r') as tagfile:
        for tag in tagfile:
            tags.append(tag.rstrip())
    return(tags)

def conllSeqGenerator(input_file):
    """ return an instance generator for a filename

    The generator yields lists of words and tags.  For test data, the tags
    may be unknown.  For usage, see trainClassifier and applyClassifier below.

    """
    unk = 'unk'
    cur_words = []
    cur_tags = []
    with open(input_file) as instances:
        for line in instances:
            if len(line.rstrip()) == 0:
                if len(cur_words) > 0:
                    yield cur_words,cur_tags
                    cur_words = []
                    cur_tags = []
            else:
                parts = line.rstrip().split()
                cur_words.append(parts[0])
                if len(parts)>1:
                    cur_tags.append(parts[1])
                else: cur_tags.append(unk)
        if len(cur_words)>0: yield cur_words,cur_tags


def train_perceptron(train_file, tags):
	weight_list = [] # weight list for perceptron
	for _ in tags:
		weight_list.append(defaultdict(int))
	
	for i in range(30):
		for _word,_tag in conllSeqGenerator(train_file):
			for i in range(len(_word)):
				word = _word[i]
				tag = _tag[i]
				max_tag = -1
				max_value = -99999
				
				for i in range(len(weight_list)):
					if weight_list[i][word] > max_value:
						max_value = weight_list[i][word]
						max_tag = tags[i]
				
				if max_tag != tag:
					weight_list[ tags.index(tag) ][word] += 1
					weight_list[ tags.index(max_tag) ][word] -= 1
	return weight_list

def train_perceptron_with_feature(train_file, tags):
	weight_list = [] # weight list for perceptron
	for _ in tags:
		weight_list.append(defaultdict(int))
	
	for i in range(30):
		for _word,_tag in conllSeqGenerator(train_file):
			for i in range(len(_word)):
				if i==0:
					_word[i] = _word[i].lower()
				word = _word[i]
				tag = _tag[i]
				max_tag = -1
				max_value = -99999
				
				for i in range(len(weight_list)):
					if weight_list[i][word] > max_value:
						max_value = weight_list[i][word]
						max_tag = tags[i]
				
				if max_tag != tag:
					weight_list[ tags.index(tag) ][word] += 1
					weight_list[ tags.index(max_tag) ][word] -= 1
	return weight_list

def calculate_dev1(dev_file, tags, weight_list):
	for _word,_tag in conllSeqGenerator(dev_file):
		for i in range(len(_word)):
			word = _word[i]
			tag = _tag[i]
			max_tag = -1
			max_value = -99999
			for weight in weight_list:
				if weight[word] > max_value:
					max_value = weight[word]
					max_tag = weight_list.index(weight)
			if word == 'the':
				print 'D'
			elif word[0] == '#':
				print '#'
			elif word[0] == '@':
				print '@'
			elif word.find('http://') != -1:
				print 'U'
			elif word[:2] == 'un':
				print 'A'
			elif word[-4:] == 'able':
				print 'A'
			elif word[-2:] == 'ly':
				print 'R'
			else:
				print tags[max_tag]
		print ''


def train_bayers(train_file, tags):
	# get word count for each tag
	weight_list = []
	for _ in tags:
		weight_list.append(defaultdict(int))

	for _word, _tag in conllSeqGenerator(train_file):
		for i in range(len(_word)):
			word = _word[i]
			tag = _tag[i]
			weight_list[tags.index(tag)][word] += 1
			for weight in weight_list:
				if word not in weight:
					weight[word] = 0
	poster_list = [ sum(weight.values()) for weight in weight_list]
	poster_list_percent = [ math.log(poster*1.0/sum(poster_list)) for poster in poster_list] 

	vocab_size = len(weight_list[0])
	alpha = 0.01

	for weight,poster in zip(weight_list,poster_list):
		for word in weight:
			weight[word] = math.log((weight[word]+alpha)*1.0/(poster+vocab_size*alpha))
		
	return weight_list,poster_list_percent

def calculate_dev(dev_file, tags, weight_list, poster_list_percent):
	for _word,_tag in conllSeqGenerator(dev_file):
		for i in range(len(_word)):
			word = _word[i]
			tag = _tag[i]
			max_tag = -1
			max_value = -99999
			for weight,poster in zip(weight_list,poster_list_percent):
				value = weight[word] + poster
				if value > max_value:
					max_value = value
					max_tag = weight_list.index(weight)
			print tags[max_tag]
		print ''

def getBestRoute(word, tags, tags_len,logPT, logPE):
	parents = [[0]*tags_len for x in xrange(len(word)-1)]
	old_value = [0]*tags_len
	new_value = [0]*tags_len

	for x in range(len(word)):
		#initial value from start
		if x == 0:
			for i in range(len(old_value)):
				old_value[i] += logPT['Start'][ tags[i] ]
				old_value[i] += logPE[ tags[i] ][word[x]]
		#bottom up iteration for all links
		else:
			for i in range(len(new_value)):
				temp = [ j+logPT[ tags[old_value.index(j)] ][ tags[i] ] for j in old_value]
				new_value[i] = max(temp)
				parents[x-1][i] = temp.index(new_value[i])
				new_value[i] += logPE[tags[i]][word[x]]
			for i in range(len(new_value)): 
				old_value[i] = new_value[i]
	#last word to end
	for i in range(len(new_value)):
		new_value[i] += logPT[tags[i]]['END']

	max_value = max(new_value)
	for i in range(len(new_value)):
		#deal with same max value situation
		#print the route
		if new_value[i] == max_value:
			route = [i]
			for item in reversed(parents):
				route.append(item[route[-1]])
			route = [tags[i] for i in route]
			route.reverse()
			print route
	print new_value

def Q1(tags):
	weight_list = train_perceptron('oct27.train',tags)
	calculate_dev1('oct27.dev', tags, weight_list)

def Q2(tags):
	weight_list,poster_list_percent = train_bayers('oct27.train', tags)
	calculate_dev('oct27.dev', tags, weight_list, poster_list_percent)

def Q3(tags):
	weight_list = train_perceptron_with_feature('oct27.train',tags)
	calculate_dev1('oct27.words.test', tags, weight_list)

def Q7():	
	word = 'They can can can can can can fish'
	word = [item.lower() for item in word.rstrip().split()]
	tags = ['N', 'V', 'Start', 'End'] 
	tags_len = len(tags)-2
	logPE = { 'N':{'they':-1 ,'can':-3, 'fish':-3 },
			  'V':{'they':-10, 'can':-2, 'fish': -3 }
			}

	logPT = {	'N':{'N': -5, 'V': -2, 'END': -2},
			  	'V':{'N': -1, 'V': -4, 'END': -3},
			  	'Start':{'N': -1, 'V': -1}
			}
	getBestRoute(word, tags, tags_len,logPT, logPE)
	
def train_hmm_PE(train_file, tags):
	alpha = 0.00001
	word_count = {}
	for tag in tags:
		word_count[tag] = defaultdict(int)
	for _word, _tag in conllSeqGenerator(train_file):
		for i in range(len(_word)):
			word = _word[i]
			tag = _tag[i]
			word_count[tag][word] += 1
	total = []
	for tag in tags:
		total.append(sum(word_count[tag].values()))

	for tag in tags:
		for i,j in word_count[tag].items():
			word_count[tag][i] = math.log((j*1.0+alpha)/(total[tags.index(tag)]+10000*alpha) )

	#if zero - 10
	return word_count

def train_hmm_PT(train_file, tags):
	alpha = 0.00001
	tran_count = {}
	for tag in tags:
		tran_count[tag] = defaultdict(int)
	tran_count['Start'] = defaultdict(int)
	for _word, _tag in conllSeqGenerator(train_file):
		#first word connect to start last word connect to end
		tran_count['Start'][_tag[0]] += 1
		for i in range(len(_tag)-1):
			tran_count[_tag[i]][_tag[i+1]] += 1
		tran_count[_tag[-1]]['END'] += 1

	total = []
	for tag in tags:
		total.append(sum(tran_count[tag].values()))
	total.append(sum(tran_count['Start'].values()))
	for i,j in tran_count['Start'].items():
		tran_count['Start'][i] = math.log((j*1.0+alpha)/(total[-1])+10000*alpha)
	for tag in tags:
		for i,j in tran_count[tag].items():
			tran_count[tag][i] = math.log((j*1.0+alpha)/(total[tags.index(tag)]+10000*alpha) )
	# if zero then - 10
	return tran_count

def dev_hmm(logPT, logPE, tags, dev_file):
	thred = -10
	tags_len = len(tags)
	for word, tag in conllSeqGenerator(dev_file):
		#print word, tag
		parents = [[0]*tags_len for x in xrange(len(word)-1)]
		old_value = [0]*tags_len
		new_value = [0]*tags_len

		for x in range(len(word)):
			#initial value from start
			if x == 0:
				for i in range(len(old_value)):
					if logPT['Start'][tags[i]] == 0:
						logPT['Start'][tags[i]] = thred
					if logPE[tags[i]][word[x]] == 0:
						logPE[ tags[i] ][tags[i]] = thred
					old_value[i] += logPT['Start'][ tags[i] ]
					old_value[i] += logPE[ tags[i] ][word[x]]
			#bottom up iteration for all links
			else:
				for i in range(len(new_value)):
					for j in old_value:
						if logPT[tags[old_value.index(j)]][tags[i]] == 0:
							logPT[tags[old_value.index(j)]][tags[i]] = thred
					temp = [ j+logPT[ tags[old_value.index(j)] ][ tags[i] ] for j in old_value]
					new_value[i] = max(temp)
					parents[x-1][i] = temp.index(new_value[i])
					if logPE[tags[i]][word[x]] == 0:
						logPE[tags[i]][word[x]] = thred
					new_value[i] += logPE[tags[i]][word[x]]
				for i in range(len(new_value)): 
					old_value[i] = new_value[i]
		#last word to end
		for i in range(len(new_value)):
			if logPT[tags[i]]['END'] == 0:
				logPT[tags[i]]['END'] = thred
			new_value[i] += logPT[tags[i]]['END']

		max_value = max(new_value)
		route = [new_value.index(max_value)]
		for item in reversed(parents):
			route.append(item[route[-1]])
		route = [tags[i] for i in route]
		route.reverse()
		for item in route:
			print item
		print ''


def dev_hmm_structured(logPT, logPE, tags, dev_file):
	tags_len = len(tags)
	for word, tag in conllSeqGenerator(dev_file):
		#print word, tag
		parents = [[0]*tags_len for x in xrange(len(word)-1)]
		old_value = [0]*tags_len
		new_value = [0]*tags_len

		for x in range(len(word)):
			#initial value from start
			if x == 0:
				for i in range(len(old_value)):
					old_value[i] += logPT['Start'][ tags[i] ]
					old_value[i] += logPE[ tags[i] ][word[x]]
			#bottom up iteration for all links
			else:
				for i in range(len(new_value)):
					temp = [ j+logPT[ tags[old_value.index(j)] ][ tags[i] ] for j in old_value]
					new_value[i] = max(temp)
					parents[x-1][i] = temp.index(new_value[i])
					new_value[i] += logPE[tags[i]][word[x]]
				for i in range(len(new_value)): 
					old_value[i] = new_value[i]
		#last word to end
		for i in range(len(new_value)):
			new_value[i] += logPT[tags[i]]['END']

		max_value = max(new_value)
		route = [new_value.index(max_value)]
		for item in reversed(parents):
			route.append(item[route[-1]])
		route = [tags[i] for i in route]
		route.reverse()
		for item in route:
			print item
		print ''

def Q9(tags):
	logPE = train_hmm_PE('oct27.train', tags)
	logPT = train_hmm_PT('oct27.train', tags)
	dev_hmm(logPT, logPE, tags, 'oct27.train')

def train_structured_perceptron_PE(train_file, tags):
	word_count = {}
	for tag in tags:
		word_count[tag] = defaultdict(int)
	for _word, _tag in conllSeqGenerator(train_file):
		for i in range(len(_word)):
			word = _word[i]
			tag = _tag[i]
			word_count[tag][word] += 1

	#if zero - 10
	return word_count

def train_structured_perceptron_PT(train_file, tags):
	tran_count = {}
	for tag in tags:
		tran_count[tag] = defaultdict(int)
	tran_count['Start'] = defaultdict(int)
	for _word, _tag in conllSeqGenerator(train_file):
		#first word connect to start last word connect to end
		tran_count['Start'][_tag[0]] += 1
		for i in range(len(_tag)-1):
			tran_count[_tag[i]][_tag[i+1]] += 1
		tran_count[_tag[-1]]['END'] += 1

	# if zero then - 10
	return tran_count

def Q11(tags):
	logPE = train_structured_perceptron_PE('oct27.train',tags)
	logPT = train_structured_perceptron_PT('oct27.train',tags)
	dev_hmm_structured(logPT, logPE, tags, 'oct27.dev')

def main():
	tags = getTags() # all the tags 
	#Q1(tags)
	#Q2(tags)
	Q3(tags)
	#Q7()
	#Q9(tags)
	#Q11(tags)
	# for weight in weight_list:
	# 	print weight

main()

