import numpy as np
import tensorflow as tf

class Model:

    def __init__(self,config,dropout_keep, vocab2int, tag2int):
        
        self.lr = config.learning_rate
        self.batch_size = config.batch_size
        self.max_len = config.max_length

        self.rnn_cell = config.rnn_size
        
        self.vocab_size = len(vocab2int)
        self.embedding_size = config.embedding_size
        self.tag_size = len(tag2int)

        self.dropout_keep = dropout_keep

        self.input_data = tf.placeholder(tf.int32, shape=[self.batch_size,self.max_len], name="input_data") 
        self.labels = tf.placeholder(tf.int32,shape=[self.batch_size,self.max_len], name="labels")
        self.input_length = tf.placeholder(tf.int32,shape=[self.batch_size], name="length")

        with tf.variable_scope("bilstm") as scope:
            self._build_net()

    def _build_net(self):
        word_embeddings = tf.get_variable("word_embeddings",[self.vocab_size, self.embedding_size])

        input_embedded = tf.nn.embedding_lookup(word_embeddings, self.input_data)
        input_embedded = tf.nn.dropout(input_embedded,self.dropout_keep)

        lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(self.rnn_cell, forget_bias=1.0, state_is_tuple=True)
        lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(self.rnn_cell, forget_bias=1.0, state_is_tuple=True)

        (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, 
                                                                         lstm_bw_cell, 
                                                                         input_embedded,
                                                                         dtype=tf.float32,
                                                                         time_major=False,
                                                                         scope=None)

        bilstm_out = tf.concat([output_fw, output_bw], axis=2)  #[batch_size, max_len, 2*rnn_cell]

        # Fully connected layer.
        W = tf.get_variable(name="W", shape=[self.batch_size,2 * self.rnn_cell, self.tag_size],
                        dtype=tf.float32)

        b = tf.get_variable(name="b", shape=[self.batch_size, self.max_len, self.tag_size], dtype=tf.float32,
                        initializer=tf.zeros_initializer())

        bilstm_out = tf.tanh(tf.matmul(bilstm_out, W) + b)    #[batch_size, max_len, tag_size]
        print("<-----bilstm_out------>",bilstm_out)

        self.out_sequence = tf.argmax(tf.nn.softmax(bilstm_out), axis= -1)
        print("<-----out_sequence------>",self.out_sequence)   #[batch_size, max_len]
         
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=bilstm_out,labels=self.labels)
        self.loss = tf.reduce_mean(self.loss) 
        print("<-----self.loss------>",self.loss)   #[batch_size, max_len]          
        

        # Training ops.
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.loss)


    def train(self, sess, batch, config):
        feed_dict = {self.input_data : np.array(batch.inputs),
                     self.labels : np.array(batch.targets),
                     self.input_length : np.array(batch.targets_length)}

        _,loss,pred= sess.run([self.train_op,self.loss,self.out_sequence], feed_dict=feed_dict)
        return loss,pred

    def test(self, sess, batch, config):
        feed_dict2 = {self.input_data : np.array(batch.inputs),
                     self.labels : np.array(batch.targets),
                     self.input_length : np.array(batch.targets_length)}

        loss,pred= sess.run([self.loss,self.out_sequence], feed_dict=feed_dict2)
        return loss,pred