import tensorflow as tf
import numpy as np
import sys
import time
from ilp import Ilp
from util import Util
from evaluation import Evaluate


class BiLstm(object):

	def __init__(self,args,data,ckpt_path): #seq_len,xvocab_size, label_size,ckpt_path,pos_size,type_size,data
		self.opt = args
		self.num_steps = 120
		self.num_class = 2
		self.word_num = data.word_size
		self.ckpt_path=ckpt_path
		self.pos_size=data.pos_size
		self.type_size=data.type_size
		self.util= Util()
		sys.stdout.write('Building Graph ')
		self._build_model(args,embedding_matrix=data.pretrained)
		sys.stdout.write('graph built\n')
		self.eval=Evaluate()

	def _build_model(self,flags,embedding_matrix):
		tf.reset_default_graph()
		tf.set_random_seed(123)
		self.input=tf.placeholder(shape=[None,self.num_steps], dtype=tf.int64)
		self.length = tf.placeholder(shape=[None,], dtype=tf.int64)
		self.pos=tf.placeholder(shape=[None,self.num_steps], dtype=tf.int64)
		self.type=tf.placeholder(shape=[None,self.num_steps], dtype=tf.int64)
		self.target = [tf.placeholder(shape=[None, ], dtype=tf.int64, name='li_{}'.format(t)) for t in   range(self.num_steps)]
		self.weight = [tf.placeholder(shape=[None, ], dtype=tf.float32, name='wi_{}'.format(t)) for t in    range(self.num_steps)]
		self.keep_prob = tf.placeholder(tf.float32)  # drop out

		if embedding_matrix is not None:
			self.embedding = tf.Variable(embedding_matrix, trainable=True, name="emb",dtype=tf.float32)#
		else:
			self.embedding = tf.get_variable("emb", [self.word_num, self.emb_dim])
		self.inputs_emb = tf.nn.embedding_lookup(self.embedding, self.input)
		if flags.use_tree:
			pos_embedding = tf.get_variable('pos_embed', [self.pos_size, 40], dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=1e-4))
			type_embedding = tf.get_variable('type_embed', [self.type_size, 40], dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=1e-4))
			pos_inputs = tf.nn.embedding_lookup(pos_embedding, self.pos)
			type_inputs = tf.nn.embedding_lookup(type_embedding, self.type)
			self.inputs_emb = tf.concat(2, [self.inputs_emb, pos_inputs,type_inputs])


		cell = tf.nn.rnn_cell.LSTMCell(num_units=flags.hidden_size, state_is_tuple=True)
		dropout_cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
		stacked_cell= tf.nn.rnn_cell.MultiRNNCell([dropout_cell] * self.opt.num_layers, state_is_tuple=True)
		outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked_cell,cell_bw=stacked_cell,dtype=tf.float32,sequence_length=self.length,inputs=self.inputs_emb)
		output_fw, output_bw = outputs
		output=	tf.concat(2, [output_fw,output_bw])
		soft_dim=self.opt.hidden_size*2
		self.softmax_w = tf.get_variable("softmax_w", [soft_dim, self.num_class])
		self.softmax_b = tf.get_variable("softmax_b", [self.num_class])
		output=tf.reshape(output,[-1,soft_dim])
		self.logits = tf.matmul(output, self.softmax_w) + self.softmax_b
		self.decode_outputs_test = tf.nn.softmax(self.logits)
		self.decode_outputs_test=tf.reshape(self.decode_outputs_test,[-1,self.num_steps,self.num_class])
		#states_fw, states_bw = states
		self.classify_out=tf.reshape(self.logits,[-1,self.num_steps,self.num_class])
		self.logits= tf.transpose(self.classify_out, [1, 0, 2])
		self.logits=tf.unpack(self.logits,axis=0)
		self.loss = tf.nn.seq2seq.sequence_loss(self.logits, self.target, self.weight, self.num_class)
		self.train_op = tf.train.AdamOptimizer(learning_rate=self.opt.learn_rate).minimize(self.loss)


	'''Training and Evaluation'''
	def train(self, data, sess=None):
		saver = tf.train.Saver()
		if not sess:
			sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=2)) # create a session
			sess.run(tf.global_variables_initializer()) 		# init all variables
		sys.stdout.write('\n Training started ...\n')
		best_loss=100
		best_epoch=0
		t1=time.time()
		for i in range(self.opt.epochs):
			try:
				loss,_=self.run_epoch(sess,data,data.train,True)
				val_loss,pred= self.run_epoch(sess, data,data.valid,False)
				t2=time.time()
				print('epoch:%2d \t time:%.2f\tloss:%f\tvalid_loss:%f'%(i,t2-t1,loss,val_loss))
				t1=time.time()
				if val_loss<best_loss:
					saver.save(sess, self.ckpt_path + self.opt.model_name + '.ckpt')
					best_loss=val_loss
					best_epoch=i
				sys.stdout.flush()
			except KeyboardInterrupt:  # this will most definitely happen, so handle it
				print('Interrupted by user at iteration {}'.format(i))
				self.session = sess
				return sess
		print('best valid accuary:%f\tbest epoch:%d'%(best_loss,best_epoch))

	# prediction
	def predict(self, data, sess):
		_, predicts = self.run_epoch(sess, data, data.test, False)
		if self.opt.use_ilp:
			pred = self.ilp_solution(predicts , data.test['weight'], data.test['length'], data.test['dfather'], data.test['dtype'])
		else:
			pred= np.argmax(predicts, axis=2)

		acc, f1, pratio, gratio=self.eval.values(pred,data.test['target'],data.test['weight'])
		print('accuary:%f,f1:%f,pratio:%f,gratio:%f' %(acc,f1,pratio,gratio))

	def run_epoch(self, sess, data,data_type,is_train):
		losses = []
		num_batch=data.gen_batch_num(data_type)
		predicts=None
		for i in range(num_batch):
			input, target, weight, length, pos, dtype,dfather,sent,compressed=data.gen_batch(data_type, i)
			if is_train:
				feed_dict = self.get_feed(input, target, weight, length, pos, dtype, keep_prob=0.8)
				_, loss_v, predict = sess.run([self.train_op, self.loss, self.decode_outputs_test], feed_dict)
			else:
				feed_dict = self.get_feed(input, target, weight, length, pos, dtype, keep_prob=1.)
				loss_v, predict= sess.run([self.loss, self.decode_outputs_test], feed_dict)
			losses.append(loss_v)
			if predicts is None:
				predicts = predict
			else:
				predicts = np.concatenate((predicts, predict))
		return np.mean(losses),predicts

	def ilp_solution(self,predict,batchW,batchL,fathers,types):
		Myilp = Ilp()
		pred_label = predict[:, :, 1]
		pred = []
		batchW_temp = np.array(batchW,copy=True)
		for j in range(pred_label.shape[0]):
			size = sum(batchW_temp[j] == 1)  #
			curr_label = pred_label[j][:size]
			curr_fathers = [int(f) for f in fathers[j]]
			curr_fathers.insert(0, 0)
			dep_length = self.util.caculate_length(curr_fathers)
			dep_length=dep_length[1:]
			curr_types = types[j][:]
			saved_types = self.util.get_typelist(curr_fathers, curr_types)
			_, retained, values = Myilp.solve_ilp_problem(size, curr_label, dep_length=dep_length, parents=curr_fathers,saved=saved_types)
			values.extend([0] * (120 - len(values)))
			pred.append(values)
		pred = np.array(pred)
		return pred

	def restore_last_session(self):
		saver = tf.train.Saver()
		sess = tf.Session()  # create a session
		saver.restore(sess, self.ckpt_path + self.opt.model_name + '.ckpt')
		print('model restored')
		return sess

	def get_feed(self, input, target, weight, length, pos, dtype, keep_prob):
		feed_dict={self.input:input}
		feed_dict.update({self.target[t]: target[t] for t in range(self.num_steps)})
		feed_dict.update({self.weight[t]: weight[t] for t in range(self.num_steps)})
		feed_dict[self.pos]=pos
		feed_dict[self.type]=dtype
		feed_dict[self.length]=length
		feed_dict[self.keep_prob] = keep_prob  # dropout prob
		return feed_dict















