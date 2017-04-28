# -*- coding: utf-8 -*-
# @Time    : 2017/2/17 16:47
import numpy as np
class Evaluate(object):
	def __init__(self):
		pass

	def values(self,pred,ground,batchW):

		pred = pred.flatten()
		ground =ground.flatten()
		batchW = batchW.flatten()
		pred = pred[batchW != 0]
		ground = ground[batchW != 0]
		f1_retain=self.caculate_f1(pred,ground,1)
		pequal_num=np.sum(pred==ground)
		ptotal=ground.size
		acc=pequal_num/float(ptotal)
		prtained=np.sum(pred==1)
		pratio=prtained/float(ptotal)

		gtained=np.sum(ground==1)
		gratio=gtained/float(ptotal)

		#accuray, f1 value of retaining, f1 value of deletion,predict compress ratio, ground truth compress ratio
		return acc,f1_retain,pratio,gratio

	def caculate_f1(self,pred,ground,label):
		#label equal to 0 or 1, 0 means deletion f1, 1 means retaining f1
		equal_num = np.sum((pred == ground) & (ground == label))
		pred_sum = np.sum(pred == label)
		ground_sum = np.sum(ground == label)
		if pred_sum==0:
			return 0
		precise = equal_num / float(pred_sum)
		recall = equal_num / float(ground_sum)
		if precise+recall==0:
			return 0
		f1 = (2 * precise * recall) / (precise + recall)
		return f1

