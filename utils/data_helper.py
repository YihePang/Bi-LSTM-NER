import numpy as np
import random
import json
import sys
import re


def load_dict(filename):   
	loadfile = open(filename,"r")         
	load_dict = {}                                     
	for line in loadfile:
		load_dict = json.loads(line)
	loadfile.close()
	return load_dict

def load_list(filename):   
	data = []
	with open(filename, 'r',encoding='UTF-8') as f:
		for line in f:
			a_data = json.loads(line)
			data.append(a_data)
	return data


def seg_char(sent):
	"""    把句子按字分开，不破坏英文结构    """    
	pattern = re.compile(r'([\u4e00-\u9fa5])')    
	chars = pattern.split(sent)    
	chars = [w for w in chars if len(w.strip())>0]    
	return chars


def index_of_sub_list(s1, s2):
	#在S1中查找S2(list类型)
	for i in range(len(s1)):
		if s1[i] == s2[0]:
			j = i
			if (j + len(s2) - 1) <= len(s1):
				for k in range(len(s2)):
					if s1[j] != s2[k]:
						break
						j = j + 1
					else:
						return i
			else:
				break
	return -1


def text2id(inputs,dicts):
	"""
	inputs: text的list
	dicts:  text2int
	"""
	id_list = []
	for token in inputs:
			if dicts.get(token):
				id_list.append(dicts.get(token)) 
	return id_list


def get_tag_list(sentence_list, one_mentions):
	#========================BIEOS标签法==================================================
	"""
	sentence_list: ['《', '时', '空', '恋', '人', '》', '失', '去', '-', '在', '线', '漫', '画']
	one_mentions:  ['时空恋人', '失去', '漫画']
	return:        ['O', 'B', 'I', 'I', 'E', 'O', 'B', 'E', 'O', 'O', 'O', 'B', 'E']
	"""
	tags = ['O'] * len(sentence_list)

	for e in range(len(one_mentions)):    #一个实体
		e_list = seg_char(one_mentions[e])
		e_len = len(e_list)#实体词的长度
		
		if e_len == 1: # 一词实体
			if e_list[0] in sentence_list:
				e_idx = index_of_sub_list(sentence_list,e_list)
				# e_idx = sentence_list.index(e_list[0])
				tags[e_idx] = 'S'

		elif e_len == 2: # 二词实体
			if e_list[0] in sentence_list:
				e_begin_idx = index_of_sub_list(sentence_list,e_list)
				# e_begin_idx = sentence_list.index(e_list[0])
				e_end_idx = e_begin_idx + e_len -1
				tags[e_begin_idx] = 'B'
				tags[e_end_idx] = 'E'

		else:    # 三词及以上实体
			if e_list[0] in sentence_list:
				e_begin_idx = index_of_sub_list(sentence_list,e_list)
				# e_begin_idx = sentence_list.index(e_list[0])
				e_end_idx = e_begin_idx + e_len - 1
				I_tags = ['I'] * (e_len - 2)
				tags[e_begin_idx] = 'B'
				tags[e_end_idx] = 'E'
				tags[e_begin_idx+1:e_end_idx] = I_tags
	return tags                       #标签


def tag2entity(sentence_list, tags):
	"""
	sentence_list: ['《', '时', '空', '恋', '人', '》', '失', '去', '-', '在', '线', '漫', '画']
	tags:          ['O', 'B', 'I', 'I', 'E', 'O', 'B', 'E', 'O', 'O', 'O', 'B', 'E']
	return :       extract_entity: ['时空恋人', '失去', '漫画']
	"""
	extract_entity = []
	for g in range(len(tags)):  #标签抽取出实体
		if tags[g] == 'B':
			one_e = []
			for k in range(g,len(tags)):
				if tags[k] == 'O':
					break
				else:
					one_e.append(sentence_list[k])
			one_str = ''.join(one_e)		
			extract_entity.append(one_str)
	# print("extract_entity:",extract_entity)
	return extract_entity


def data2id(data, vocab2int, tag2int):
	"""
	data:
	[
	   one: [ [sentence_id],[tags_id] ],
	   one: [ [sentence_id],[tags_id] ],
	   ……
	]
	"""
	datas = []	
	for i in range(len(data)):
		one = []
		one_data = data[i]
		# print(one_data['text_id'])
		sentence = one_data['text']  #纯文本
		
		sentence_list = seg_char(sentence)         
		# print(sentence_list)         #分字后的sentence_list-------------------------(1)

		one.append(text2id(sentence_list,vocab2int)) #-------------1  sentence_id

		mentions_list = one_data['mention_data']
		one_mentions = []
		for j in range(len(mentions_list)):
			one_mentions.append(mentions_list[j]['mention'])	
		#print(one_mentions)      #实体 文本
		
		tags = get_tag_list(sentence_list, one_mentions)   #根据实体生成标签---------------------（2）

		one.append(text2id(tags,tag2int)) #----------------2    tag_id

		if len(text2id(tags,tag2int)) != len(text2id(sentence_list,vocab2int)): #check
			print("---errors-----")

		datas.append(one)
	return datas
		


class Batch:  #一个batch数据的类
	def __init__(self):
		self.inputs = []
		self.inputs_length = []
		self.targets = []
		self.targets_length = []


def batching_and_padding(samples,max_length):  
	padToken = 0
	batch = Batch()

	for sample in samples:	
		if len(sample[0]) < max_length:
			source = sample[0]
			pad = [padToken] * (max_length - len(source))
			# inputs
			batch.inputs.append(source + pad)         
			batch.inputs_length.append(len(source))
			#targets
			batch.targets.append(sample[1] + pad) 
			batch.targets_length.append(len(sample[1]))   

		elif len(sample[0]) >= max_length:
			source = sample[0]                       
			# inputs
			batch.inputs.append(source[:max_length])
			batch.inputs_length.append(max_length)
			#targets
			batch.targets.append(sample[1][:max_length]) 
			batch.targets_length.append(max_length) 
	
	return batch


def Batches_data(data_id, batch_size,max_length): #all datas 进行batch
	random.shuffle(data_id)
	batches = []
	data_len = len(data_id)
	def genNextSamples():
		batch_nums = int(data_len/batch_size)  
		for i in range(0, batch_nums*batch_size, batch_size):
			yield data_id[i: i + batch_size]
		if data_len % batch_size != 0:   
			last_num = data_len - batch_nums*batch_size
			up_num = batch_size - last_num
			l1 = data_id[batch_nums*batch_size : data_len]
			l2 = data_id[0: up_num]
			yield l1+l2
	for samples in genNextSamples(): 
		batch = batching_and_padding(samples,max_length)   #zero-padding
		batches.append(batch)
	
	return batches  #(batch类的list)
			



#--------------------------调用--------------------------------------------------	
"""
data = load_list("../data/dev.json")
print(len(data))  

vocab2int = load_dict("../data/dict/vocab2int.json")
int2vocab = load_dict("../data/dict/int2vocab.json")

tag2int = load_dict("../data/dict/tag2int.json")
int2tag = load_dict("../data/dict/int2tag.json")


ss = data2id(data, vocab2int, tag2int)
print(len(ss))
print(ss[0][0])
print(ss[0][1])


bs = Batches_data(ss, 50 ,100)
print(len(bs))
print(len(bs[0].inputs))          #[50,100]
print(len(bs[0].inputs_length))   #[50]
print(len(bs[0].targets))         #[50,100]
print(len(bs[0].targets_length))  #[50]
"""