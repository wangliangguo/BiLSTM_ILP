# -*- coding: utf-8 -*-
# @Time    : 2017/1/22 14:52
class Util(object):
	def __init__(self):
		pass

	def caculate_length(self, fathers):
		dep_length = []
		for i in range(len(fathers)):
			if i == 0:
				dep = 0
				dep_length.append(dep)
				continue  # for bos
			dep = 1
			fa = fathers[i]
			while fa != 0:
				dep += 1
				fa = fathers[fa]
			dep_length.append(dep)
		maxl = max(dep_length)
		dep_length = [dl / float(maxl) for dl in dep_length]
		return dep_length

	def get_typelist(self, fathers, types):
		saved_type=[u'mwe', u'expl', u'auxpass', u'neg', u'aux', u'det', u'poss', u'cop', u'pobj', u'nsubjpass', u'number', u'quantmod', u'pcomp', u'num', u'nsubj']
		saved = []
		for i, type in enumerate(types):
			if type in saved_type:
				saved.append([i, fathers[i], type])
		return saved

