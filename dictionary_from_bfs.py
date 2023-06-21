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
#from igraph import *
import matplotlib.pyplot as plt
from queue import Queue
import pandas as pd
import generate_original_graph_for_input


def positive_node_and_path_finder(adj_list, start_node, target_node,graph):  # input - the target node(start_node) and the 
                                                                      # list of nodes reachable from the target nodes.
                                                                      # list of nodes reachable from the target nodes is the output
                                                                      # of BFS algorithm
                                                                      # Output - path and the set of nodes which postively influnce the target node
    target_node1=target_node                            
    # Set of visited nodes to prevent loops
    visited = set()
    queue = Queue()

    # Add the start_node to the queue and visited list
    queue.put(start_node)
    visited.add(start_node)
    
    # start_node has not parents
    parent = dict()
    parent[start_node] = None

    # Perform step 3
    path_found = False
    while not queue.empty():
        current_node = queue.get()
        if current_node == target_node:
            path_found = True
            break
        if(current_node in adj_list):
            for next_node in adj_list[current_node]:
                if next_node not in visited:
                    queue.put(next_node)
                    parent[next_node] = current_node
                    visited.add(next_node)
                         
                
    # Path reconstruction
    path = []
    if path_found:
        path.append(target_node)
        while parent[target_node] is not None:
            path.append(parent[target_node]) 
            target_node = parent[target_node]
        path.reverse()
    return path        
    #print("")
    # Here path is the list of nodes example   [80674, 299259, 9321243, 101694462]
    # now need to find if the last node of the path is posstively being influenced if yes the return the path
    # consisting of events id
    #number_of_positively_activated=0
    #number_of_negatively_activated=0
    '''if(len(path)>3):
        pass
    positively_activated_node=[]
    negatively_activated_node=[]
    for i in range(len(path)-1):
        
        previous_node=path[i]
        next_node=path[i+1]
        sum_positive=0
        sum_negative=0
        for j in graph[previous_node][next_node]:
            if(j[2]=='pos'):
                sum_positive=j[1]+sum_positive
            if(j[2]=='neg'):
                sum_negative=j[1]+sum_negative
        if(sum_positive>sum_negative):
            positively_activated_node.append(previous_node)
        else:
            negatively_activated_node.append(previous_node)
            
    if(len(positively_activated_node) > len(negatively_activated_node)):
        target_node=[]
        target_node.append(target_node1)
        return (target_node,path) # we only need the positive node
    else:
        target_node1=[]
        return(target_node1,path)'''    # Return no node since number of negatively activated node > number of positively activated node

   

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

def removekey(d, key):
    r = dict(d)
    del r[key]
    return r      
    
from collections import Counter   

def shortest_path_dictionary_generator(intermediate_graph,dict_for_target_users,S):
    dict1={}
    target_users=set()
    intermediate_graph_copy=intermediate_graph
    for member in dict_for_target_users:
        target_users.add(member)
        
    for target_user in target_users:
        visited=[]
        intermediate_graph=intermediate_graph_copy               
        list1=bfs(visited, intermediate_graph, target_user) # List 1 should be the list of seed users
        list1.remove(target_user)
        temp={}
        for influencial_member in S:
            
            intermediate_graph=intermediate_graph_copy
            list2=[]  # will contain list of path
            if(influencial_member not in list1):
                continue
            #here will be loop which will continue until no shortest path found
            iteration=1
            while(1):
                path = positive_node_and_path_finder(intermediate_graph, target_user, influencial_member,intermediate_graph)
                if(iteration ==1):
                    shortest_path=len(path)
                iteration=iteration+1
                
                if(len(path)>0) and (len(path)==shortest_path):  # path found
                    
                    list2.append(path) 
                    #path_copy=path
                    # Now remove the middle vertices of the path from intermediate_graph
                    #path_copy.remove(target_user)
                    #path_copy.remove(influencial_member)
                    path = filter(lambda x: x != target_user, path)
                    path = filter(lambda x: x != influencial_member, path)
                    path=list(path)
                    if(len(path)>0):
                        list_of_nodes_to_be_removed=[]
                        for individual_nodes in path:
                            list_of_nodes_to_be_removed.append(individual_nodes)
                            #intermediate_graph.pop(individual_nodes, None)
                            intermediate_graph=removekey(intermediate_graph, individual_nodes)
                    if(len(path)==0):
                        break
                        # remove the list_of_nodes from intgermediate graph    
                    #target_user=path[0]
                    #influencial_user=path[len(path)-1]
                else:
                    break
            if(len(list2)>0):        
                temp[influencial_member]=list2
                    #print("")
        if ( bool(temp)):        
            dict1[target_user]=temp
        
                    
                    
                
    return  dict1   
    
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
def generating_many_list_of_shortest_path(G,number_of_graphs,dict_for_target_users,S):
    #number_of_graphs=100
    list_of_dictionry_of_shortest_path=[]
    '''list_of_intermediate_graph=[]
    for no in range(number_of_graphs):
        
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
        # Here we get the intermediate graph
        list_of_intermediate_graph.append(intermediate_graph)'''
    dictionary_for_shortest_path=shortest_path_dictionary_generator(G,dict_for_target_users,S)        
    list_of_dictionry_of_shortest_path.append(dictionary_for_shortest_path)
    return  G,list_of_dictionry_of_shortest_path      


def generating_many_list_of_shortest_path_part1(G,number_of_graphs,dict_for_target_users,S):
    number_of_graphs=100
    list_of_dictionry_of_shortest_path=[]
    list_of_intermediate_graph=[]
    for no in range(number_of_graphs):
        
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
        # Here we get the intermediate graph
        list_of_intermediate_graph.append(intermediate_graph)
        dictionary_for_shortest_path=shortest_path_dictionary_generator(intermediate_graph,dict_for_target_users,S)        
        list_of_dictionry_of_shortest_path.append(dictionary_for_shortest_path)
    return  list_of_intermediate_graph,list_of_dictionry_of_shortest_path 



'''def generating_many_list_of_shortest_path_from_intermediate_graph(G,p,number_of_graphs,dict_for_target_users,S,list_of_intermediate_graph):
    list_of_dictionry_of_shortest_path=[]
    
    for graph1 in list_of_intermediate_graph:
        
           
        # Here we get the intermediate graph
        
        dictionary_for_shortest_path=shortest_path_dictionary_generator(graph1,dict_for_target_users,S)        
        list_of_dictionry_of_shortest_path.append(dictionary_for_shortest_path)
    return  list_of_intermediate_graph,list_of_dictionry_of_shortest_path'''   
#list_of_intermediate_graph,list_of_dictionry_of_shortest_path=generating_many_list_of_shortest_path(reverse_graph,50,dict_for_target_users,S) 
#top_influencial_member=find_top_influencial_member(final_list_of_RRS_set,20) # 20 is the budget 
#print(top_influencial_member)
#print("")