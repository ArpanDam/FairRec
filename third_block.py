# -*- coding: utf-8 -*-
"""
Created on Wed May 31 10:51:27 2023

@author: HP
"""

import pickle
import numpy as np
from collections import Counter  
import random
from random import uniform, seed
import numpy as np
import Evaluation2
from queue import Queue
def func1(seed,best_tag,graph):
    # Find the graph having only the best_tag
    dict1={}
    for inf_member in graph:
        temp={}
        for follower in graph[inf_member]:
            list1=[]
            for i in graph[inf_member][follower]:
                event_topic=i[0]
                event_probability=i[1]
                
                event_sign=i[2]
                if event_topic in best_tag:
                    
                    list1.append((event_topic,event_probability,event_sign))
            if(len(list1)>0):        
                temp[follower]=list1
        if(len(temp)>0):        
            dict1[inf_member]=temp 
    graph=dict1
    return graph            

def func2(seed,graph):
    tag=set()
    for inf_member in graph:
        if inf_member in seed:
            for follower in graph[inf_member]:
                for i in graph[inf_member][follower]:
                    event_topic=i[0]
                    tag.add(event_topic)
    return tag                    




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
    #print("")
    # Here path is the list of nodes example   [80674, 299259, 9321243, 101694462]
    # now need to find if the last node of the path is posstively being influenced if yes the return the path
    # consisting of events id
    #number_of_positively_activated=0
    #number_of_negatively_activated=0
    if(len(path)>3):
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
    '''if(len(negatively_activated_node)>0):
        print("")'''        
    if(len(positively_activated_node) > len(negatively_activated_node)):
        target_node=[]
        target_node.append(target_node1)
        return (target_node,path) # we only need the positive node
    else:
        target_node1=[]
        return(target_node1,path)
    
    
    
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
                
def RRS_set_genetaor(graph,dict_for_target_users):
    target_users=set()
    for member in dict_for_target_users:
        target_users.add(member)
    list_of_RRS_set=[]
          
        
    for target_user in target_users:
        visited=[]               
        list1=bfs(visited, graph, target_user)
        list1.remove(target_user)
        list_of_RRS_set.append(list1)
        set_positive_nodes=set()
        for member in list1:
            positive_node,path = positive_node_and_path_finder(graph, target_user, member,graph)
            if(len(positive_node)>0):
                for i in positive_node:
                    set_positive_nodes.add(i)
        if(len(set_positive_nodes)>0):
            list_of_RRS_set.append(list(set_positive_nodes))
    return(list_of_RRS_set)                      
            
def genrating_many_sampled_graph(G,number_of_graphs,dict_for_target_users):
    final_list_of_RRS_set=[]
    #list_of_intermediate_graph=[]
    #number_of_graphs=500
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
        #list_of_intermediate_graph.append(intermediate_graph)        
        RRS_set=RRS_set_genetaor(intermediate_graph,dict_for_target_users) # get multiple RRS set for each intermediate graoh       
        for individual_RRS_set in RRS_set:
            final_list_of_RRS_set.append(individual_RRS_set)
            
        #print("")
    return  final_list_of_RRS_set          
    
def find_top_influencial_member(RRS_set,k): # k is the budget of the influencial member
    SEED=[]
    for _ in range(k):
        
        # Find node that occurs most often in R and add to seed set
        flat_list = [item for sublist in RRS_set for item in sublist]
        seed = Counter(flat_list).most_common()[0][0]
        #print(seed)
        SEED.append(seed)
        
        # Remove RRSs containing last chosen seed 
        RRS_set = [rrs for rrs in RRS_set if seed not in rrs]
        
        # Record Time
        #timelapse.append(time.time() - start_time)
    return((SEED))   

'''def smallest_group():


def largest_group():


def desparity(): '''

def disparity(influenced_member,reverse_graph):
    dict1={}  # key tags, values = node id
    for follower in reverse_graph:
        if follower in influenced_member:
            # Find the tag
            for inf in reverse_graph[follower]:
                for i in reverse_graph[follower][inf]:
                    event_topic=i[0]
                    if i[0] not in dict1:
                        list1=[]
                        list1.append(follower)
                        dict1[i[0]]=list1
                    else:
                        list1=dict1[i[0]]
                        list1.append(follower)
                        dict1[i[0]]=list1
    dict1_sum={}
    for key in dict1:
        length=len(dict1[key])
        dict1_sum[key]=length  
    dict1_sum={k: v for k, v in sorted(dict1_sum.items(), key=lambda item: item[1])}  
   
    #disparity= dict1_sum[dict1_sum.keys()[-1]] - dict1_sum[dict1_sum.keys()[0]] 
    #("Disparity = ",disparity)                     
    return dict1_sum                        
    
    
def least_influenced_seed(seed,top_influencial_member):
    print("seeds are",seed)
    dict_seed_rank={}
    index=0
    for i in top_influencial_member:
        if i in seed:
            dict_seed_rank[i]=index
            
        index=index+1 
    dict_seed_rank={k: v for k, v in sorted(dict_seed_rank.items(), key=lambda item: item[1])} 
    #print(dict_seed_rank)  
    return dict_seed_rank
    #return list(dict_seed_rank.keys())[-1]      
    
    
