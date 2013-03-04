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

def char_match(strg, search=re.compile(r'[^a-z]').search):
	return not bool(search(strg))



vocab = {}
allword = Set([])
priorpolarity = ['positive', 'negative']

positive = 'POS'
negative = 'NEG'
objective = 'OBJ'
tag = [positive, negative, objective]
file_trainkey = 'train.key'
threshold = 7
symmetric_para = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
#get vocab hashtable of priorpolarity of each word
filename = 'sentiment-vocab.tff'
document = open(filename)
for line in document:
	a = wordpunct_tokenize(line)
	vocab[a[8]] = a[-1]


#get priorpolarity of each file 
path = 'dev/'
listing = os.listdir(path)
filelist = []


for filename in listing:
	document = open(path+filename)
	dic = defaultdict(int)
	for line in document:
		for word in wordpunct_tokenize(line):
			word = word.lower()
			if char_match(word):
				dic[word] += 1
				allword.add(word)
	filelist.append(dic)

priorlist = []
for document in filelist:
	poscount = 0
	negcount = 0
	for word in document.items():
		if word[0] in vocab:
			if vocab[word[0]] == priorpolarity[0]: #pos
				poscount += word[1]
			elif vocab[word[0]] == priorpolarity[1]:
				negcount += word[1]
	if poscount - negcount > threshold:
		priorlist.append(positive)
	elif negcount - poscount > threshold:
		priorlist.append(negative)
	else: 
		priorlist.append(objective)

#for item in priorlist:
#	print item


#Bayers
#get from training file

label = {'POS':0, 'NEG':1, 'OBJ':2}
P_label = [0,0,0]

training_priorpolarity = []
with open(file_trainkey,'r') as keyfile:
	for keyline in keyfile:
		training_priorpolarity.append(wordpunct_tokenize(keyline)[-1])

for item in training_priorpolarity:
	P_label[label[item]] += 1

P_label = [item*1.0/sum(P_label) for item in P_label ]



#list of three label default dic
pos_dic = defaultdict(int)
neg_dic = defaultdict(int)
obj_dic = defaultdict(int)
dic_list = [pos_dic, neg_dic, obj_dic]

for dic in dic_list:
	for word in vocab:
		dic[word] = 0


path = 'train/'
listing = os.listdir(path)
index = 0

for filename in listing:
	document = open(path+filename)
	for line in document:
		for word in wordpunct_tokenize(line):
			word = word.lower()
			if char_match(word):
			    dic = dic_list[ label[ training_priorpolarity[index] ] ]
			    dic[word] += 1
			for dic in dic_list:
				if word not in dic:
					dic[word] = 0
	index += 1


def get_total(dic):
	total = 0
	for word in dic:
		total += dic[word]
	return total

dic_list2 = copy.deepcopy(dic_list)

for dic in dic_list:
	total = get_total(dic)
	#test
	#print total
	for word in dic:
		dic[word] = (dic[word]*1.0+symmetric_para[4])*1.0/(total+symmetric_para[4]*len(dic))
		#get log
		if dic[word] > 0:
			dic[word] = math.log(dic[word])

# for i in pos_dic.values():
# 	print i 
	

#Q5
# X = [i for i in range(len(pos_dic))]
# C = obj_dic.values()

# plt.scatter(X, C)

# plt.savefig("5 obj 6.pdf")

#Q5 part2
# big_list6 = pos_dic.values()+neg_dic.values()+obj_dic.values()

# for dic in dic_list2:
# 	total = get_total(dic)
# 	#test
# 	#print total
# 	for word in dic:
# 		dic[word] = (dic[word]*1.0+symmetric_para[0])*1.0/(total+symmetric_para[0]*len(dic))
# 		#get log
# 		if dic[word] > 0:
# 			dic[word] = math.log(dic[word])
# big_list = dic_list2[0].values() + dic_list2[1].values() + dic_list2[2].values()


# mean0 = sum(big_list) / float(len(big_list))
# mean6 = sum(big_list6) / float(len(big_list6))
# ss0 = sum((mean0-i)**2 for i in big_list)
# ss6 = sum((mean6-i)**2 for i in big_list6)
# ss06 = sum( (big_list[i]-mean0) * (big_list6[i]-mean6) for i in range(len(big_list)))
# R2 = ss06**2/ss0
# R2 = R2/ss6
# print R2

#Q6
# Y = neg_dic.values()
# Y2 = pos_dic.values()
# Y3 = [Y[i] - Y2[i] for i in range(len(Y))]
# for i in range(len(Y3)):
# 	if Y3[i] > 4.499:
# 		print pos_dic.keys()[i]

# Y3.sort()
# print Y3
# Y2 = neg_dic.values()
# Y = pos_dic.values()
# Y3 = [Y2[i] - Y[i] for i in range(len(Y))]
# for i in range(len(Y3)):
#  	if Y3[i] > 14.02:
#  		print i

#Y3.sort()
#print Y3
#print neg_dic
P_label = [math.log(item) for item in P_label ]
#print P_label

#print P_label
#print dic_list

#test dev file
path = 'test/'
listing = os.listdir(path)
priorlist = []

for filename in listing:
	document = open(path+filename)
	#word list in document
	word_list = defaultdict(int)
	for line in document:
		for word in wordpunct_tokenize(line):
			word = word.lower()
			if char_match(word):
				word_list[word] += 1
	#print word_list
	#try three labels
	max_score = -99999
	max_index = -1
	for i in range(3):
		score = P_label[i]
		for word in word_list:
			score += dic_list[i][word] * word_list[word]
		#print score
		if score > max_score:
			max_score = score
			max_index = i

	priorlist.append(tag[max_index])

for item in priorlist:
 	print item



# X = symmetric_para
# C = [0.18, 0.175, 0.167, 0.16, 0.131, 0.105, 0.105]
# S = [0.03, 0.02, 0.01, 0.0049, 0.0024, 0.0012, 0.0012]
# plt.plot(X, C)
# plt.plot(X, S)

# plt.savefig("a.pdf")



