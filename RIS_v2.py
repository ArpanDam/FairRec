# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 15:08:04 2023

@author: HP
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 17:06:50 2022

@author: HP
"""

import pickle
import random
from random import uniform, seed
import numpy as np
#from igraph import *
import pickle
#import matplotlib.pyplot as plt
#from scipy import stats
from numpy import save
from numpy import load
from numpy import dot
from numpy.linalg import norm

import matplotlib.pyplot as plt
from queue import Queue
import pandas as pd
import generate_original_graph_for_input


def generate_graph_having_only_r_tags(reverse_graph,dict_for_influence_tags):
    influence_tags=set()
    dict1={}
    for tags in dict_for_influence_tags:
        influence_tags.add(tags)
    for follower in reverse_graph:
        temp={}
        for influencer in reverse_graph[follower]:
            
            list1=[]
            for i in reverse_graph[follower][influencer]:
                if (i[0] in influence_tags):
                    list1.append((i[0],i[1],i[2]))
            if (len(list1)>0):        
                temp[influencer]=list1
        if ( bool(temp)): # false if empty        
            dict1[follower]=temp
    return dict1         
                


   

def bfs(visited, graph, node): #function for BFS , output the set of nodes reachable from the input node
  visited = [] # List for visited nodes.
  queue = []     #Initialize a queue
  visited.append(node)
  queue.append(node)
  list1=[]   
  while queue:          # Creating loop to visit each node
    m = queue.pop(0) 
    #print (m, end = " ")
    list1.append(m)
    
    if m in graph:
        for neighbour in graph[m]:
            if neighbour not in visited:
                visited.append(neighbour)
                queue.append(neighbour)
    else:
        pass             
                
  return list1

def RRS_set_genetaor(graph,dict1):
    
    target_users=set()
    for member in dict1:
        target_users.add(member)
    list_of_RRS_set=[]
          
        
    for target_user in target_users:
        visited=[]               
        list1=bfs(visited, graph, target_user)
        list1.remove(target_user)
        if(len(list1)>0):
            list_of_RRS_set.append(list1)
        '''set_positive_nodes=set()
        for member in list1:
            positive_node,path = positive_node_and_path_finder(graph, target_user, member,graph)
            if(len(positive_node)>0):
                for i in positive_node:
                    set_positive_nodes.add(i)
        if(len(set_positive_nodes)>0):
            list_of_RRS_set.append(list(set_positive_nodes))'''
    return(list_of_RRS_set)            
    
from collections import Counter   

#RRS_set=RRS_set_genetaor(filtered_reverse_graph,dict_for_target_users)

# Find the influencial members having most number of occusrence in RRS_set
def find_top_influencial_member(RRS_set,k): # k is the budget of the influencial member
    SEED=[]
    for _ in range(k):
        
        # Find node that occurs most often in R and add to seed set
        flat_list = [item for sublist in RRS_set for item in sublist]
        seed = Counter(flat_list).most_common()[0][0]
        SEED.append(seed)
        
        # Remove RRSs containing last chosen seed 
        RRS_set = [rrs for rrs in RRS_set if seed not in rrs]
        
        # Record Time
        #timelapse.append(time.time() - start_time)
    return(sorted(SEED))        
def genrating_many_sampled_graph(G,number_of_graphs,dict1,node_topic):
    # for each of the key in dict1 generate number_of_graphs
    all_rrs_set=[]
    dict_topic_rrs={}
    for key in dict1: # each key means different topic
        # find the node_id in node_topic
        node_topic_id=set()
        for key2 in node_topic:
            if node_topic[key2]==key:
                node_topic_id.add(key2)
    
        final_list_of_RRS_set=[]
        #list_of_intermediate_graph=[]
        #number_of_graphs=500
        
        for no in range(number_of_graphs): # for each attribute  gebnerate 
            
            intermediate_graph={}
            for follower in G:
                temp={}
                for influencer in G[follower]:
                    list1=[]
                    for i in G[follower][influencer]:
                        # i correspond to each edge
                        if(random.uniform(0, 1)<i[1]): # add the edge else continue
                            list1.append((i[0],i[1],i[2]))
                    if(len(list1)>0):        
                        temp[influencer]=list1
                if ( bool(temp)):        
                    intermediate_graph[follower]=temp 
            #list_of_intermediate_graph.append(intermediate_graph)        
            RRS_set=RRS_set_genetaor(intermediate_graph,node_topic_id) # get multiple RRS set for each intermediate graoh       
            for individual_RRS_set in RRS_set:
                all_rrs_set.append(individual_RRS_set)
                final_list_of_RRS_set.append(individual_RRS_set)
        dict_topic_rrs[key]=final_list_of_RRS_set
        final_list_of_RRS_set=[]
        #print("")
    return  dict_topic_rrs, all_rrs_set   
   