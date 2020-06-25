import tensorflow as tf
import numpy as np 
import math
import os
import random

from model import Model
from utils.config import get_config
from utils.data_helper import load_dict,load_list,data2id,Batches_data
from check import call_results,call_results2,ana_results,ana_results2, results_by_text_len

#配置参数
config = get_config()

#加载数据
dev_data = load_list(config.dev_data_file)
print("loading....dev_data...finish!.....lengths: ",len(dev_data))

vocab2int = load_dict(config.vocab2int_file)
int2vocab = load_dict(config.int2vocab_file)
tag2int = load_dict(config.tag2int_file)
int2tag = load_dict(config.int2tag_file)

#数据转换
dev_data = data2id(dev_data, vocab2int, tag2int)


tf.set_random_seed(config.random_seed)
with tf.Session() as sess:
	with tf.variable_scope("build_model"):
		train_model = Model(config=config, dropout_keep=1, vocab2int=vocab2int, tag2int=tag2int)
	with tf.variable_scope("build_model",reuse=True ):
		test_model =  Model(config=config, dropout_keep=1, vocab2int=vocab2int, tag2int=tag2int)

	model_saver = tf.train.Saver(max_to_keep=1)   #model 保存,初始化

	#是否已有模型
	ckpt = tf.train.get_checkpoint_state(config.model_dir)  #读取check_point
	#加载已有的模型
	if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
		print('Reloading model parameters..')
		model_saver.restore(sess, ckpt.model_checkpoint_path)   #加载保存的模型
		current_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
		current_step = int(current_step)
		print("current_step:",current_step)
	else:
		print('find no model')
		current_step = 0	

	dev_batches = Batches_data(dev_data, config.batch_size, config.max_length)	

	previous_f1 = 0.0


	#----dev---
	t_pred = []     #所有测试的结果
	t_label = []     #所有测试
	t_loss = np.array([])
	for t_batch in dev_batches:   #一个batch的训练数据
		t_label.append(t_batch)
		test_loss,test_pred = test_model.test(sess, t_batch, config)
		t_pred.append(test_pred)
		t_loss = np.append(t_loss,test_loss)
	
	test_avg_loss = np.mean(t_loss, axis=0)
	test_P,test_R,test_F = call_results2(t_label,t_pred,int2vocab,int2tag)
	print("----- TEST -- Loss %.2f --------precision %.3f-----recall %.3f------f1 %.3f" % (test_avg_loss,test_P,test_R,test_F))
	
	print("-------------------实体词长度分析----------------------------------------------------")
	#2 
	p1,r1,f1,n1 = ana_results2(t_label,t_pred,int2vocab,int2tag,2)
	print("Len = 2: precision %.3f-----recall %.3f------f1 %.3f------nums  %d" % (p1,r1,f1,n1))

	#3
	p1,r1,f1,n1 = ana_results2(t_label,t_pred,int2vocab,int2tag,3)
	print("Len = 3: precision %.3f-----recall %.3f------f1 %.3f------nums  %d" % (p1,r1,f1,n1))

	#4
	p1,r1,f1,n1 = ana_results2(t_label,t_pred,int2vocab,int2tag,4)
	print("Len = 4: precision %.3f-----recall %.3f------f1 %.3f------nums  %d" % (p1,r1,f1,n1))

	#5
	p1,r1,f1,n1 = ana_results2(t_label,t_pred,int2vocab,int2tag,5)
	print("Len = 5: precision %.3f-----recall %.3f------f1 %.3f------nums  %d" % (p1,r1,f1,n1))

	#6
	p1,r1,f1,n1 = ana_results2(t_label,t_pred,int2vocab,int2tag,6)
	print("Len = 6: precision %.3f-----recall %.3f------f1 %.3f------nums  %d" % (p1,r1,f1,n1))

	#7
	p1,r1,f1,n1 = ana_results2(t_label,t_pred,int2vocab,int2tag,7)
	print("Len = 7: precision %.3f-----recall %.3f------f1 %.3f------nums  %d" % (p1,r1,f1,n1))

	#8
	p1,r1,f1,n1 = ana_results2(t_label,t_pred,int2vocab,int2tag,8)
	print("Len = 8: precision %.3f-----recall %.3f------f1 %.3f------nums  %d" % (p1,r1,f1,n1))

	#9
	p1,r1,f1,n1 = ana_results2(t_label,t_pred,int2vocab,int2tag,9)
	print("Len = 9: precision %.3f-----recall %.3f------f1 %.3f------nums  %d" % (p1,r1,f1,n1))

	#10
	p1,r1,f1,n1 = ana_results2(t_label,t_pred,int2vocab,int2tag,10)
	print("Len = 10: precision %.3f-----recall %.3f------f1 %.3f------nums  %d" % (p1,r1,f1,n1))
	
	# 10以上
	p1,r1,f1,n1 = ana_results(t_label,t_pred,int2vocab,int2tag, 11,100)
	print("11 <= Len < 100: precision %.3f-----recall %.3f------f1 %.3f------nums  %d" % (p1,r1,f1,n1))

	# 10以上
	p1,r1,f1,n1 = ana_results(t_label,t_pred,int2vocab,int2tag, 10,100)
	print("10 <= Len < 100: precision %.3f-----recall %.3f------f1 %.3f------nums  %d" % (p1,r1,f1,n1))


	print("-------------------句子长度分析----------------------------------------------------")
	p1,r1,f1 = results_by_text_len(t_label,t_pred,int2vocab,int2tag, 0,10)
	print("0 <= Len < 10: precision %.3f-----recall %.3f------f1 %.3f------" % (p1,r1,f1))

	p1,r1,f1 = results_by_text_len(t_label,t_pred,int2vocab,int2tag, 10,20)
	print("10 <= Len < 20: precision %.3f-----recall %.3f------f1 %.3f-----" % (p1,r1,f1))

	p1,r1,f1 = results_by_text_len(t_label,t_pred,int2vocab,int2tag, 20,30)
	print("20 <= Len < 30: precision %.3f-----recall %.3f------f1 %.3f------" % (p1,r1,f1))

	p1,r1,f1 = results_by_text_len(t_label,t_pred,int2vocab,int2tag, 30,100)
	print("30 <= Len < 100: precision %.3f-----recall %.3f------f1 %.3f-----" % (p1,r1,f1))
	
	
	
