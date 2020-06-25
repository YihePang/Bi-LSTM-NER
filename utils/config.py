import tensorflow as tf
import numpy as np 

# 模型参数
# 用于支持接受命令行传递参数
#第一个是参数名称，第二个参数是默认值，第三个是参数描述
tf.app.flags.DEFINE_integer('rnn_size', 200, 'Number of hidden units in each layer')
tf.app.flags.DEFINE_integer('num_layers', 1, 'Number of layers in each encoder and decoder')
tf.app.flags.DEFINE_integer('embedding_size', 300, 'Embedding dimensions of encoder and decoder inputs')

tf.app.flags.DEFINE_integer('max_length', 50, 'max_source_length')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')
tf.app.flags.DEFINE_integer('batch_size', 50, 'Batch size')
tf.app.flags.DEFINE_integer('numEpochs', 6, 'Maximum # of training epochs')
tf.app.flags.DEFINE_integer('random_seed', 1234, 'random seed')

tf.app.flags.DEFINE_integer('steps_per_checkpoint', 1, 'Save model checkpoint every this iteration')

tf.app.flags.DEFINE_string('model_dir', 'model/', 'Path to save model checkpoints')
tf.app.flags.DEFINE_string('model_name', 'train_model.ckpt', 'File name used for model checkpoints')


tf.app.flags.DEFINE_string('int2vocab_file', 'data/dict/int2vocab.json','pass of dict')
tf.app.flags.DEFINE_string('vocab2int_file', 'data/dict/vocab2int.json','pass of dict')

tf.app.flags.DEFINE_string('int2tag_file', 'data/dict/int2tag.json','pass of dict')
tf.app.flags.DEFINE_string('tag2int_file', 'data/dict/tag2int.json','pass of dict')

tf.app.flags.DEFINE_string('train_data_file', 'data/train1.json','pass of dict')
tf.app.flags.DEFINE_string('dev_data_file', 'data/dev.json','pass of dict')
tf.app.flags.DEFINE_string('test_data_file', 'data/test.json','pass of dict')

# tf.app.flags.DEFINE_string('pre_word_emb', 'emb/my_emb.json','pass of dict')

FLAGS = tf.app.flags.FLAGS
# print(FLAGS.model_name)  #调用参数


class Config:  
	def __init__(self):
		self.rnn_size = FLAGS.rnn_size
		self.num_layers = FLAGS.num_layers
		self.embedding_size = FLAGS.embedding_size
		
		self.max_length = FLAGS.max_length
		self.learning_rate = FLAGS.learning_rate
		self.batch_size = FLAGS.batch_size
		self.numEpochs = FLAGS.numEpochs
		self.random_seed = FLAGS.random_seed

		self.steps_per_checkpoint = FLAGS.steps_per_checkpoint
		
		self.model_dir = FLAGS.model_dir
		self.model_name = FLAGS.model_name

		self.int2vocab_file = FLAGS.int2vocab_file
		self.vocab2int_file = FLAGS.vocab2int_file

		self.int2tag_file = FLAGS.int2tag_file
		self.tag2int_file = FLAGS.tag2int_file

		self.train_data_file = FLAGS.train_data_file
		self.dev_data_file = FLAGS.dev_data_file
		self.test_data_file = FLAGS.test_data_file


def get_config():
	config = Config()
	return config