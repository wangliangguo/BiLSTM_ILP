# -*- coding: utf-8 -*-
# @Time    : 2017/1/19 10:48
import pulp
from pulp import *
import sys

class Ilp(object):
	def __init__(self):
		pass


	def solve_ilp_problem(self,word_num,p,dep_length=None,parents=None,matrix=None,solver='glpk',mx=0.7,mn=0.2,w2=0.5,saved=None):
		max_length=int(mx*word_num)
		min_length=int(mn*word_num)
		prob = pulp.LpProblem('sentence_compression', pulp.LpMaximize)
		# initialize the word binary variables
		c = pulp.LpVariable.dicts(name='c',indexs=range(word_num),lowBound=0,upBound=1,cat='Integer')
		#objective function
		prob += sum( (p[i]-w2*dep_length[i])*c[i] for i in range(word_num)) #p[i] is  the probability for retain the words
		# #constraints
		if parents is not None:
			for j in range(word_num):
				if parents[j]>0:
					prob +=c[j]<=c[parents[j]]
		if saved is not None:
			for s in saved:
				prob+=c[s[0]]>=c[s[1]]
		prob+=sum([c[i] for i in range(word_num)])<=max_length
		prob+=sum([c[i] for i in range(word_num)])>=min_length
		if solver == 'gurobi':
			prob.solve(pulp.GUROBI(msg=0))
		elif solver == 'glpk':
			prob.solve(pulp.GLPK(msg=0))
		elif solver == 'cplex':
			prob.solve(pulp.CPLEX(msg=0))
		else:
			sys.exit('no solver specified')
		values=[c[j].varValue for j in range(word_num)]
		solution = [j for j in range(word_num) if c[j].varValue == 1]
		return (pulp.value(prob.objective), solution,values)





