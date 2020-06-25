import tensorflow as tf
import numpy as np 
import math
import os
import random

from model import Model
from utils.config import get_config
from utils.data_helper import load_dict,load_list,data2id,Batches_data
from check import call_results,call_results2

#配置参数
config = get_config()

#加载数据
train_data = load_list(config.train_data_file)
print("loading....train_data...finish!.....lengths: ",len(train_data))
dev_data = load_list(config.dev_data_file)
print("loading....dev_data...finish!.....lengths: ",len(dev_data))

vocab2int = load_dict(config.vocab2int_file)
int2vocab = load_dict(config.int2vocab_file)
tag2int = load_dict(config.tag2int_file)
int2tag = load_dict(config.int2tag_file)

#数据转换
train_data = data2id(train_data, vocab2int, tag2int)
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
	#训练完，加载已有的模型
	if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
		print('Reloading model parameters..')
		model_saver.restore(sess, ckpt.model_checkpoint_path)   #加载保存的模型
		current_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
		current_step = int(current_step)
		print("current_step:",current_step)

	#新模型，开始训练
	else:
		print('Created new model parameters..')
		sess.run(tf.global_variables_initializer())
		current_step = 0	

	train_batches = Batches_data(train_data, config.batch_size, config.max_length)
	dev_batches = Batches_data(dev_data, config.batch_size, config.max_length)	

	previous_f1 = 0.0

	for e in range(config.numEpochs):
		random.shuffle(train_batches)
		#print("one_epoch_steps:",len(train_batches))
		print("----- Epoch {}/{} -----".format(e + 1, config.numEpochs))
		#train_data
		for batch in train_batches:    #循环  一个batch的训练数据
			current_step += 1	
			train_loss,train_pred = train_model.train(sess, batch, config) 
			
			if current_step % config.steps_per_checkpoint == 0:
				train_P,train_R,train_F = call_results(batch,train_pred,int2vocab,int2tag)
				print("----- Step %d -- Loss %.2f --------precision %.3f-----recall %.3f------f1 %.3f" % (current_step,train_loss,train_P,train_R,train_F))
				
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
				


				out_file = open('result.txt', 'a')
				if current_step == config.steps_per_checkpoint: #参数写入
					out_file.write('rnn_size %d; rnn_size %d; embedding_size %d; max_length %d; learning_rate_base %.3f; batch_size %d; ' 
									% (config.rnn_size, config.num_layers, config.embedding_size,
										config.max_length, config.learning_rate, config.batch_size))
					out_file.write('\n')
				out_file.write("----- Step %d -- Loss %.2f --------precision %.3f-----recall %.3f------f1 %.3f" % (current_step,train_loss,train_P,train_R,train_F))
				out_file.write('\n')
				out_file.write("----- TEST -- Loss %.2f --------precision %.3f-----recall %.3f------f1 %.3f" % (test_avg_loss,test_P,test_R,test_F))
				out_file.write('\n')
				

				if test_F > previous_f1:  #只保存最好f值的模型
					model_saver.save(sess,config.model_dir, global_step=current_step)  #model 保存
					previous_f1 = test_F
