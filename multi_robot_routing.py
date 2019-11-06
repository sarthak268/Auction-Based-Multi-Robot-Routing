import copy
import math
from collections import defaultdict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

agents = {}
prices = {}
unassigned = []
values_matrix = []
averages = []

path_followed = []

agent_locations = [(0, 0), (4, 1)]
target_locations = [(1, 1), (1, 3), (2, 4), (3, 2), (4, 4)]

def visualize(path):
	edges = []
	for i in range(len(agent_locations)):
		edges.append([])

	agent_num = 0
	cost = 0
	for i in range(len(path)):
		if (i != 0):
			if (path[i][0][0] in agent_locations):
				agent_num += 1
	
		begin = path[i][0][0]
		end = path[i][1][0]
		cost += path[i][2]
		edges[agent_num].append([begin, end])

	print ('Total cost of routing = ' + str(cost))

	color = ['r', 'b', 'g', 'black', 'yellow']
		
	for i in range(len(edges)):
		for j in range(len(edges[i])):
			b = edges[i][j][0]
			e = edges[i][j][1]
			plt.plot([b[0], e[0]], [b[1], e[1]], c=color[i])
			if (b in agent_locations):
				plt.plot(b[0], b[1], marker='+', c='black')
			else:
				plt.plot(b[0], b[1], marker='o', c='black')
	plt.ylim(-2, 6)
	plt.xlim(-2, 6)
	plt.savefig('trajectory.png')
	plt.close()

class route:
	def __init__(self, num_agents, num_targets, agent_loc, target_loc, unallocated_targets = [], cost_function=None):
		global path_followed

		self.num_agents = num_agents
		target_indices = {}
		for i in range(len(target_loc)):
			target_indices[target_loc[i]] = i + num_agents

		unallocated_targets = target_loc.copy()
		self.agents_targets = defaultdict(list)
		
		for i in range(num_agents):
			self.agents_targets[i] = []

		final_path = []

		while(len(unallocated_targets)>0):

			min_val = float('inf')
			index_dict = {}
			parent_agent = -1

			curr_target = unallocated_targets.pop(0)
			
			print('Adding Target ... ', curr_target)
			for agent in self.agents_targets.keys():

				self.vertices = 2 + len(self.agents_targets[agent])

				targets = self.agents_targets[agent].copy()
				targets.append(curr_target)
				
				g = [[0 for column in range(self.vertices)] for row in range(self.vertices)]
				target_index_here = {}
				target_index_here[agent_loc[agent]] = 0
				for i in range(len(targets)):
					target_index_here[targets[i]] = i+1
				
				agent_ind = 0
				for k in self.agents_targets[agent]:
					g[agent_ind][target_index_here[k]] = self.distance(agent_loc[agent], k)
				
				g[agent_ind][agent_ind] = 0
				g[agent_ind][target_index_here[curr_target]] = self.distance(agent_loc[agent], curr_target)
				
				for i in targets:
					for j in targets:
						g[target_index_here[i]][target_index_here[j]] = self.distance(i, j)
						g[target_index_here[j]][target_index_here[i]] = self.distance(j, i)
				
				weight, p, g1 = self.primMST(g)
				#print(agent, '----', weight)
				if(weight<min_val):
					min_val = weight
					parent_agent = agent
					index_dict = target_index_here
					parent = p
					graph = g1

			self.agents_targets[parent_agent].append(curr_target)
			
			print('Dict = ', self.agents_targets)
			#print('Dict', index_dict)
			for i1 in range(1, len(graph)):
				if(parent[i1]==None):
					continue
				vertex_1 = [loc for loc, index in index_dict.items() if index == parent[i1]]
				vertex_2 = [loc for loc, index in index_dict.items() if index == i1]
			
				path = [vertex_1, vertex_2, graph[parent[i1]][i1]]
				if(path not in final_path):
					final_path.append(path)

		#print(self.agents_targets)
		print ('<<<<=================================>>>>')
		print ('FINAL PATH ==>')
		print(final_path)
		path_followed = final_path

	def distance(self, x, y):
		return (x[0]-y[0])**2 + (x[1]-y[1])**2 

	def printMST(self, parent, graph, V):
		#print ("Edge \tWeight")
		for i in range(1, V):
			if(parent[i]==None):
				continue
			#print (parent[i], "-", i, "\t", graph[parent[i]][i])
			
	def minKey(self, key, mstSet, V): 
  
		# Initilaize min value 
		min1 = float('inf')
		min_index = -1
		for v in range(V):
			if (key[v] < min1 and mstSet[v] == False): 
				min1 = key[v] 
				min_index = v
  
		return min_index

	def primMST(self, graph): 
		V = len(graph)
		key = [float('inf')] * V 
		parent = [None] * V
		key[0] = 0
		mstSet = [False] * V 
  
		parent[0] = -1
		for cout in range(V): 
  
			u = self.minKey(key, mstSet, V)
			 
			mstSet[u] = True
			for v in range(V): 
				if graph[u][v] > 0 and mstSet[v] == False and key[v] > graph[u][v]: 
					key[v] = graph[u][v] 
					parent[v] = u

		weight = 0 
		for i in range(1, V):
			if(parent[i]==None):
				continue
			weight += graph[parent[i]][i]

		#print('MST here')
		self.printMST(parent, graph, V)
		return weight, parent, graph

g = route(len(agent_locations), len(target_locations), agent_locations, target_locations)

visualize(path_followed)
