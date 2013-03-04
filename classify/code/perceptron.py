from nltk.tokenize import word_tokenize, wordpunct_tokenize, sent_tokenize
from sets import Set
import re
import os
import sys
import math
import numpy as np
import copy
import matplotlib.pyplot as plt
from collections import defaultdict

def char_match(strg, search=re.compile(r'[^a-z]').search):
	return not bool(search(strg))

vocab = {}

positive = 'POS'
negative = 'NEG'
objective = 'OBJ'
tag = [positive, negative, objective]
file_trainkey = 'train.key'

#get vocab hashtable of priorpolarity of each word
filename = 'sentiment-vocab.tff'
document = open(filename)
for line in document:
	a = wordpunct_tokenize(line)
	vocab[a[8]] = a[-1]



pos_weight = defaultdict(int)
neg_weight = defaultdict(int)
obj_weight = defaultdict(int)
weight_list = [pos_weight, neg_weight, obj_weight]

for weight in weight_list:
	for word in vocab:
		weight[word] = 0

#get training data, key

training_priorpolarity = []
with open(file_trainkey,'r') as keyfile:
	for keyline in keyfile:
		training_priorpolarity.append(wordpunct_tokenize(keyline)[-1])

path = 'train/'
listing = os.listdir(path)

#word count list
word_count = []
for filename in listing:
	document = open(path+filename)
	dic = defaultdict(int)
	for line in document:
		for word in wordpunct_tokenize(line):
			word = word.lower()
			if char_match(word):
			    dic[word] += 1
	word_count.append(dic)
aver_weight_list = copy.deepcopy(weight_list)

for i in range(30):
	num = 0
	for document in word_count:
		#get three type score and find the max
		max_score = -999
		prior = -1
		index = 0
		for weight in weight_list:
			score = 0
			for word in document:
				score += document[word]*weight[word]
			if score > max_score:
				max_score = score
				prior = index

			index += 1
		if tag[prior] != training_priorpolarity[num]:
			#add weight for prior list
			#reduce weight for training list
			for word in document:
				weight_list[prior][word] -= document[word]
				weight_list[tag.index(training_priorpolarity[num])][word] += document[word]
		num += 1
	# average perceptron
	for i in range(3):
		for item in aver_weight_list[i]:
			aver_weight_list[i][item] += weight_list[i][item]
# for i in pos_weight.values():
# 	print i
#print aver_weight_list
#Q8
# Y1 = pos_weight.values()
# Y = neg_weight.values()
# Y3 = [Y[i] - Y1[i] for i in range(len(Y))]
# for i in range(len(Y3)):
# 	if Y3[i] > 40:
# 		print pos_weight.keys()[i]

#Y3.sort()
#print Y3

def get_result(weight_list, word_count):
	#get training data result
	# for document in word_count:
	# 		#get three type score and find the max
	# 		max_score = -999
	# 		index = 0
	# 		prior = -1
	# 		for weight in weight_list:
	# 			score = 0
	# 			for word in document:
	# 				score += document[word]*weight[word]
	# 			if score > max_score:
	# 				max_score = score
	# 				prior = index
	# 			index += 1
	# 		print tag[prior]

	#get dev data result
	# path = 'dev/'
	# listing = os.listdir(path)

	# #word count list
	# word_count = []
	# for filename in listing:
	# 	document = open(path+filename)
	# 	dic = defaultdict(int)
	# 	for line in document:
	# 		for word in wordpunct_tokenize(line):
	# 			word = word.lower()
	# 			if char_match(word):
	# 			    dic[word] += 1
	# 	word_count.append(dic)

	# for document in word_count:
	# 		#get three type score and find the max
	# 		max_score = -999
	# 		index = 0
	# 		prior = -1
	# 		for weight in weight_list:
	# 			score = 0
	# 			for word in document:
	# 				score += document[word]*weight[word]
	# 			if score > max_score:
	# 				max_score = score
	# 				prior = index
	# 			index += 1
	# 		print tag[prior]
	#get test data
	path = 'test/'
	listing = os.listdir(path)

	#word count list
	word_count = []
	for filename in listing:
		document = open(path+filename)
		dic = defaultdict(int)
		for line in document:
			for word in wordpunct_tokenize(line):
				word = word.lower()
				if char_match(word):
				    dic[word] += 1
		word_count.append(dic)

	for document in word_count:
			#get three type score and find the max
			max_score = -999
			index = 0
			prior = -1
			for weight in weight_list:
				score = 0
				for word in document:
					score += document[word]*weight[word]
				if score > max_score:
					max_score = score
					prior = index
				index += 1
			print tag[prior]
get_result(aver_weight_list, word_count)

# X = [i+1 for i in range(30)]
# print X
# T = [0.7032, 0.8029, 0.8613, 0.944, 0.899, 0.9307, 0.9672, 0.9708, 0.9757, 0.9781, 0.983, 0.9818, 0.983, 0.9988, 0.9988,
# 	 0.9988, 0.9988, 0.9988, 0.9988, 0.9988, 0.9988, 0.9988, 0.9988, 0.9988, 0.9988, 0.9988, 0.9988, 0.9988, 0.9988, 0.9988]
# D = [0.6109, 0.6182, 0.6836, 0.7164, 0.6945, 0.72, 0.6836, 0.6655, 0.7455, 0.68, 0.7382, 0.7382, 0.7527, 0.7236, 0.7236,
# 	 0.7236, 0.7236, 0.7236, 0.7236, 0.7236, 0.7236, 0.7236, 0.7236, 0.7236, 0.7236, 0.7236, 0.7236, 0.7236, 0.7236, 0.7236,]
# plt.plot(X, T)
# plt.plot(X, D)

#aver perceptron
#Q9
# X = [i+1 for i in range(30)]
# T = [0.5316, 0.5925, 0.6156, 0.6277, 0.6496, 0.6655, 0.6740, 0.6764, 0.6813, 0.6800, 0.6849, 0.6934, 0.6959, 0.6971, 0.6995, 
# 		0.6983, 0.6983, 0.6995, 0.6983, 0.6995, 0.6995, 0.6995, 0.6995, 0.6995, 0.7019,0.7019,0.7007,0.7007, 0.7007,0.7007]
# D = [0.4182, 0.4582, 0.4727, 0.4655, 0.4764,0.4836, 0.48, 0.48, 0.4727, 0.4691, 0.48, 0.48, 0.48, 0.48, 0.48, 
# 	 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48]
# plt.plot(X, T)
# plt.plot(X, D)

# plt.savefig("9.pdf")

