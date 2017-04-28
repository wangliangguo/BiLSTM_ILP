# -*- coding: utf-8 -*-
# @Time    : 2017/1/12 16:04

import math
import pickle

class Data(object):
	def __init__(self,data_path,vocab_path,pretrained,batch_size):
		self.batch_size = batch_size

		data, vocab ,pretrained= self.load_vocab_data(data_path,vocab_path,pretrained)
		self.train=data['train']
		self.valid=data['valid']
		self.test=data['test']
		self.word_size = len(vocab['word2id'])
		self.pos_size  = len(vocab['pos2id'])
		self.type_size = len(vocab['type2id'])
		self.pretrained=pretrained

	def gen_batch(self,data,i):
		begin=i*self.batch_size
		data_size=data['input'].shape[0]
		end=(i+1)*self.batch_size
		if end>data_size:
			end=data_size
		input=data['input'][begin:end]
		target = data['target'][begin:end]
		length=data['length'][begin:end]
		weight=data['weight'][begin:end]
		pos=data['pos'][begin:end]
		dtype=data['dtype'][begin:end]
		dfather=data['dfather'][begin:end]
		sent=data['sent'][begin:end]
		compressed=data['compressed'][begin:end]

		return input,target.T,weight.T,length,pos,dtype,dfather,sent,compressed

	def load_vocab_data(self,data_path,vocab_path,pretrained):

		with open(data_path, 'rb') as fdata, open(vocab_path, 'rb') as fword2id:
			data = pickle.load(fdata)
			vocab = pickle.load(fword2id)

		with open(pretrained, 'rb') as fin:
			pretrained = pickle.load(fin)

		return data, vocab, pretrained


	def gen_batch_num(self,data): #
		data_size = data['input'].shape[0]
		batch_num=math.ceil(data_size/float(self.batch_size))

		return int(batch_num)

