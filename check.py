import numpy as np
import random
import json
import sys
import re

def id2text(inputs,dicts):
	"""
	inputs: text的list
	dicts:  text2int
	"""
	text_list = []
	for token in inputs:
			if dicts.get(str(token)):
				text_list.append(dicts.get(str(token))) 
	return text_list


def taglist2entity(sentence_list, tags):
	"""
	sentence_list: ['《', '时', '空', '恋', '人', '》', '失', '去', '-', '在', '线', '漫', '画']
	tags:          ['O', 'B', 'I', 'I', 'E', 'O', 'B', 'E', 'O', 'O', 'O', 'B', 'E']
	return :       extract_entity: ['时空恋人', '失去', '漫画']
	"""
	extract_entity = []
	for g in range(len(sentence_list)):  #标签抽取出实体
		if tags[g] == 'B':
			one_e = []
			for k in range(g,len(sentence_list)):
				if tags[k] == 'E':
					one_e.append(sentence_list[k])
					break
				else:
					one_e.append(sentence_list[k])
			one_str = ''.join(one_e)		
			extract_entity.append(one_str)
	# print("extract_entity:",extract_entity)
	return extract_entity


def call_results(batch,pred,int2vocab,int2tag):
	"""
	batch: class
	pred: [batch_size,max_len]
	"""
	correct_nums = 0
	pred_nums = 0
	gold_nums = 0

	batch_size  = len(pred)
	for b in range(batch_size):
		gold_entity_list = []
		pred_entity_list = []
		gold_entity_list = taglist2entity( id2text(batch.inputs[b], int2vocab), id2text(batch.targets[b], int2tag))
		pred_entity_list = taglist2entity( id2text(batch.inputs[b], int2vocab), id2text(pred[b], int2tag))	
	
		result = [1 if e in gold_entity_list else 0 for e in pred_entity_list]
		correct_nums += sum(result)
		gold_nums += len(gold_entity_list)
		pred_nums += len(pred_entity_list)

	precision = float(correct_nums)/float(pred_nums) if pred_nums > 0. else 0.
	recall = float(correct_nums)/float(gold_nums) if gold_nums > 0. else 0.
	f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0. else 0.

	return precision,recall,f1




def call_results2(batch,pred,int2vocab,int2tag):
	"""
	batch: class-list
	pred: [batch_size,max_len]
	"""
	correct_nums = 0
	pred_nums = 0
	gold_nums = 0
	
	for i in range(len(batch)):
		batch_size  = len(pred[i])
		for b in range(batch_size):
			gold_entity_list = []
			pred_entity_list = []
			gold_entity_list = taglist2entity( id2text(batch[i].inputs[b], int2vocab), id2text(batch[i].targets[b], int2tag))
			pred_entity_list = taglist2entity( id2text(batch[i].inputs[b], int2vocab), id2text(pred[i][b], int2tag))	
		
			result = [1 if e in gold_entity_list else 0 for e in pred_entity_list]
			correct_nums += sum(result)
			gold_nums += len(gold_entity_list)
			pred_nums += len(pred_entity_list)

	precision = float(correct_nums)/float(pred_nums) if pred_nums > 0. else 0.
	recall = float(correct_nums)/float(gold_nums) if gold_nums > 0. else 0.
	f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0. else 0.

	return precision,recall,f1


def selesct_l_entity(sentence_list, tags,  min_l, max_l):
	"""
	sentence_list: ['《', '时', '空', '恋', '人', '》', '失', '去', '-', '在', '线', '漫', '画']
	tags:          ['O', 'B', 'I', 'I', 'E', 'O', 'B', 'E', 'O', 'O', 'O', 'B', 'E']
	return :       extract_entity: ['时空恋人', '失去', '漫画']
	"""
	extract_entity = []
	for g in range(len(sentence_list)):  #标签抽取出实体
		if tags[g] == 'B':
			one_e = []
			for k in range(g,len(sentence_list)):
				if tags[k] == 'E':
					one_e.append(sentence_list[k])
					break
				else:
					one_e.append(sentence_list[k])
			if len(one_e) >= min_l and len(one_e) < max_l:
				one_str = ''.join(one_e)		
				extract_entity.append(one_str)
	# print("extract_entity:",extract_entity)
	entity_nums = len(extract_entity)
	return extract_entity,entity_nums


def ana_results(batch,pred,int2vocab,int2tag, min_l, max_l):
	"""
	batch: class
	pred: [batch_size,max_len]
	"""
	correct_nums = 0
	pred_nums = 0
	gold_nums = 0
	class_nums = 0
	for i in range(len(batch)):
		batch_size  = len(pred[i])
		for b in range(batch_size):
			gold_entity_list = []
			pred_entity_list = []
			gold_entity_list,g_nums = selesct_l_entity( id2text(batch[i].inputs[b], int2vocab), id2text(batch[i].targets[b], int2tag), min_l, max_l)
			pred_entity_list,p_nums = selesct_l_entity( id2text(batch[i].inputs[b], int2vocab), id2text(pred[i][b], int2tag), min_l, max_l)	
		
			result = [1 if e in gold_entity_list else 0 for e in pred_entity_list]
			correct_nums += sum(result)
			gold_nums += len(gold_entity_list)
			pred_nums += len(pred_entity_list)
			class_nums += g_nums

	precision = float(correct_nums)/float(pred_nums) if pred_nums > 0. else 0.
	recall = float(correct_nums)/float(gold_nums) if gold_nums > 0. else 0.
	f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0. else 0.

	return precision,recall,f1,class_nums


def selesct_entity(sentence_list, tags,  s_l):
	"""
	sentence_list: ['《', '时', '空', '恋', '人', '》', '失', '去', '-', '在', '线', '漫', '画']
	tags:          ['O', 'B', 'I', 'I', 'E', 'O', 'B', 'E', 'O', 'O', 'O', 'B', 'E']
	return :       extract_entity: ['时空恋人', '失去', '漫画']
	"""
	extract_entity = []

	for g in range(len(sentence_list)):  #标签抽取出实体
		if tags[g] == 'B':
			one_e = []
			for k in range(g,len(sentence_list)):
				if tags[k] == 'E':
					one_e.append(sentence_list[k])
					break
				else:
					one_e.append(sentence_list[k])
			if len(one_e) == s_l:
				one_str = ''.join(one_e)		
				extract_entity.append(one_str)
	# print("extract_entity:",extract_entity)
	entity_nums = len(extract_entity)
	return extract_entity, entity_nums


def ana_results2(batch,pred,int2vocab,int2tag, s_l):
	"""
	batch: class
	pred: [batch_size,max_len]
	"""
	correct_nums = 0
	pred_nums = 0
	gold_nums = 0
	class_nums = 0   #该类别下的实体个数（标准集)
	for i in range(len(batch)):
		batch_size  = len(pred[i])
		for b in range(batch_size):
			gold_entity_list = []
			pred_entity_list = []
			gold_entity_list,g_nums = selesct_entity( id2text(batch[i].inputs[b], int2vocab), id2text(batch[i].targets[b], int2tag), s_l)
			pred_entity_list,p_nums = selesct_entity( id2text(batch[i].inputs[b], int2vocab), id2text(pred[i][b], int2tag), s_l)	
		
			result = [1 if e in gold_entity_list else 0 for e in pred_entity_list]
			correct_nums += sum(result)
			gold_nums += len(gold_entity_list)
			pred_nums += len(pred_entity_list)
			class_nums += g_nums

	precision = float(correct_nums)/float(pred_nums) if pred_nums > 0. else 0.
	recall = float(correct_nums)/float(gold_nums) if gold_nums > 0. else 0.
	f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0. else 0.

	return precision,recall,f1,class_nums




def results_by_text_len(batch,pred,int2vocab,int2tag, min_l, max_l):
	"""
	batch: class-list
	pred: [batch_size,max_len]
	"""
	correct_nums = 0
	pred_nums = 0
	gold_nums = 0
	
	for i in range(len(batch)):
		batch_size  = len(pred[i])
		for b in range(batch_size):
			if batch[i].inputs_length[b] >= min_l and batch[i].inputs_length[b] < max_l:
				gold_entity_list = []
				pred_entity_list = []
				gold_entity_list = taglist2entity( id2text(batch[i].inputs[b], int2vocab), id2text(batch[i].targets[b], int2tag))
				pred_entity_list = taglist2entity( id2text(batch[i].inputs[b], int2vocab), id2text(pred[i][b], int2tag))	
			
				result = [1 if e in gold_entity_list else 0 for e in pred_entity_list]
				correct_nums += sum(result)
				gold_nums += len(gold_entity_list)
				pred_nums += len(pred_entity_list)

	precision = float(correct_nums)/float(pred_nums) if pred_nums > 0. else 0.
	recall = float(correct_nums)/float(gold_nums) if gold_nums > 0. else 0.
	f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0. else 0.

	return precision,recall,f1

