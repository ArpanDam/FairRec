# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 09:31:21 2023

@author: HP
"""

import pickle
import numpy as np
import generate_original_graph_for_input

#edge_probability_career=pickle.load(open("edge_probability_career","rb"))

edge_probability_career=pickle.load(open("edge_probability_career","rb")) # portion of the graph

file_name_sign="influence_tags_sign_"+str(10)
sign_dictionary=pickle.load(open("../Example of career group tags and its dictionary of positive negative tags_0.85/"+file_name_sign,"rb"))  # portion of the graph
graph=generate_original_graph_for_input.generate_final_graph(sign_dictionary,edge_probability_career)

shprtest_path_tag=pickle.load(open("shprtest_path_tag","rb"))
'''print("")
# remove the negative edges

dict1={}
# removes negative edges
for inf_member in  graph:
    temp={}
    for follower in graph[inf_member]:
        
        list1=[]
        for i in graph[inf_member][follower]:
            event_topic=i[0]
            event_probability=i[1]
            
            event_sign=i[2]
            if event_sign=='pos':
                
                list1.append((event_topic,event_probability,event_sign))
        if(len(list1)>0):        
            temp[follower]=list1
    if(len(temp)>0):        
        dict1[inf_member]=temp

graph=dict1 
dict2={}
for inf in graph:
    temp={}
    for follower in graph[inf]:
        
        
        best_tag= graph[inf][follower][0][0]
        prob=graph[inf][follower][0][1]
        for path in graph[inf][follower]:
            if(path[1]>prob):
                best_tag=path[0]
                prob=path[1]
        temp[follower]=(best_tag,prob)
    dict2[inf]=temp
      
print("") 

with open('shprtest_path_tag', 'wb') as file:
      
    # A new file will be created
    pickle.dump(dict2, file)''' 
                         
tag=set()
def find_tag(path):
    for i in range(len(path)-1):
        member1=path[i]
        member2=path[i+1]
        tag.add(shprtest_path_tag[member1][member2][0])

def find_best_path(list1):
    
    best_path=list1[0]
    best_sum=0
    for each_path in list1:
        sum1=0
        # find the probability 
        for i in range(len(each_path)-1):
            member1=each_path[i]
            member2=each_path[i+1]
            prob=shprtest_path_tag[member1][member2][1]
            sum1=sum1+prob
        if sum1>best_sum:
            best_sum=sum1
            best_path=each_path
    return best_path           
            
def tag_finder(list_of_dictionary_of_shortest_path):
    for follower in list_of_dictionary_of_shortest_path[0]:
        for inf in list_of_dictionary_of_shortest_path[0][follower]:
            list1=[]
            for path in  list_of_dictionary_of_shortest_path[0][follower][inf]:
                
                path.reverse()
                list1.append(path)
            best_path=find_best_path(list1)    
            find_tag(best_path)
    return tag           
                