# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 18:32:37 2022

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


#edge_probability_career = pickle.load(open("edge_probability_career","rb"))

#event_tags_from_all_group_attributes_merge_upto67000=pickle.load(open("../event_tags_from_all_group_attributes_merge_upto67000","rb"))
#event_participation_influence=pickle.load(open("../event_participation_influence_career_not_related_event_with_eid_u5_h=12_e=3_interested=6_with_event_tags_from_group_attributes_matching","rb"))
'''number_of_nodes=set()
for key in edge_probability_career:
    for i in edge_probability_career[key]'''


# Genreate edges having 2 features

#for member in edge_probability_career:
    



def generate_final_graph(sign_dictionary,edge_probability_career):
    dict1={} # dictionary which contains the final graph
    for inf_member in  edge_probability_career:
        temp={}
        for follower in edge_probability_career[inf_member]:
            
            list1=[]
            for i in edge_probability_career[inf_member][follower]:
                event_topic=i[0]
                event_probability=i[1]
                if(event_topic in sign_dictionary):
                    event_sign=sign_dictionary[i[0]]
                    
                    list1.append((event_topic,event_probability,event_sign))
            temp[follower]=list1
        dict1[inf_member]=temp
    return dict1

#sign_dictionary=pickle.load(open("../Example of career group tags and its dictionary of positive negative tags/influence_tags_sign_5","rb"))        
#group_tags=np.load("../Example of career group tags and its dictionary of positive negative tags/group_tags_5.npy", mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
# 15 group tags 
#graph=generate_final_graph(sign_dictionary)

#gtags_word_embeddings=pickle.load(open("../../../gtags_word_embeddings","rb"))
def check_event_tags_difference(t1,t2):
    t1_embedding=gtags_word_embeddings[t1]
    t2_embedding=gtags_word_embeddings[t2]
    den=(norm(t1_embedding)*norm(t2_embedding))
    if(den==0):
	    return 0
    else:
	    return dot(t1_embedding,t2_embedding)/den
def edge_influence_tags(graph,group_tags):
    event_tags_set=set() # event_tags which are positive
    for influencial_member in graph:
        for follower in graph[influencial_member]:
            for i in graph[influencial_member][follower]:
                if(i[2]=='pos'):
                    event_tag=i[0]
                    event_tags_set.add(event_tag)
   # Get all the event tags
    group_tags=group_tags.tolist()
    dict1={}
    for event_tag in event_tags_set:
        max_sim=0
        for group_tag in group_tags:
            if(event_tag=='Creating a Healthy Relationship to Food'):
                print("")
            similarity=check_event_tags_difference(event_tag,group_tag)
            if(similarity > max_sim):
                max_sim=similarity
                final_event_tag=group_tag
        dict1[event_tag]=final_event_tag
    # Got the dict1
    dict2={} # final graph
    for influencial_member in graph:
        temp={}
        for follower in graph[influencial_member]:
            list1=[]
            for i in graph[influencial_member][follower]:
                list1.append((dict1[i[0]],i[1],i[2]))
            temp[follower]=list1
        dict2[influencial_member]=temp    
    return dict2            
            
                    
#final_graph=edge_influence_tags(graph,group_tags) # Final graph contains the event tags compared with the group tags and assigned group tags of highest similarity               
            
            
    
# Total members is 1100
# Find the best target users let T =500 
# Get a dictionary with key user id and value number of positive indegree edges

def generate_reverse_graph(graph):
    dict1={} # reverse of graph dictionary , key = follower value=dic with key influencial member
        # and value a list of event, prob and sign
    for influencial_member in graph:
        for follower in graph[influencial_member]:
            temp={}
            list1=[] # stores the list of tuples containing event prob and sign
            for i in  graph[influencial_member][follower]:
                list1.append(i)
                    
                temp[influencial_member]=list1
                if(follower not in dict1):
                    dict1[follower]=temp
                else:
                    dict1[follower][influencial_member]=list1
                        #print("")
                    
   
    return (dict1)
#reverse_graph=generate_reverse_graph(graph)

def sort_dictionary_by_value(x): # x is the inputted dictionary
    sorted_dict={}
    sorted_dict={k: v for k, v in sorted(x.items(), reverse=True ,key=lambda item: item[1])}
    return sorted_dict  # sorted in ascending order
    
def give_target_user(reverse_graph,number_of_target_user):
    dict1={} # key target user id and value number of positive indegree edges
    for follower in reverse_graph:
        sum1=0 # number of positive indegree
        for influencial_member in reverse_graph[follower]:
            for i in reverse_graph[follower][influencial_member]:
                if(i[2]=='pos'):
                    sum1=sum1+1
        dict1[follower]=sum1
    sorted_dict1=sort_dictionary_by_value(dict1) # 840 total target members
    dict1={}
    index=1
    for follower in sorted_dict1:
        dict1[follower]=sorted_dict1[follower]
        index=index+1
        if(index == number_of_target_user):
            break
        
    return dict1

#dict_for_target_users=give_target_user(reverse_graph,400) # return the the dictionary for number of target users 


def give_best_postive_tags(dict_for_target_users,reverse_graph,r): # where r is the budget of the number of positive tags
    dict1={} # key : influence tags , value : number of occurence of influence tags
    for follower in dict_for_target_users:
        for influencial_user in  reverse_graph[follower]:
            for i in reverse_graph[follower][influencial_user]:
                event_tag=i[0]
                if(i[2]=='pos'):
                    if(event_tag in dict1):
                        number=dict1[event_tag]
                        dict1[event_tag]=number+1
                    else:
                        
                        dict1[event_tag]=1
    sorted_dict=sort_dictionary_by_value(dict1)                    
    dict1={}
    index=1
    for influence_tag in sorted_dict:
        dict1[influence_tag]=sorted_dict[influence_tag]
        index=index+1
        if(index == r):
            break
        
    return dict1                    
def give_best_negative_tags(dict_for_target_users,reverse_graph,r): # where r is the budget of the number of positive tags
    dict1={} # key : influence tags , value : number of occurence of influence tags
    for follower in dict_for_target_users:
        for influencial_user in  reverse_graph[follower]:
            for i in reverse_graph[follower][influencial_user]:
                event_tag=i[0]
                if(i[2]=='neg'):
                    if(event_tag in dict1):
                        number=dict1[event_tag]
                        dict1[event_tag]=number+1
                    else:
                        
                        dict1[event_tag]=1
    sorted_dict=sort_dictionary_by_value(dict1)                    
    dict1={}
    index=1
    for influence_tag in sorted_dict:
        dict1[influence_tag]=sorted_dict[influence_tag]
        index=index+1
        if(index == r):
            break
        
    return dict1 
#dict_for_influence_tags=give_best_postive_tags(dict_for_target_users,reverse_graph,20) # this 20 is the budget r                   
print("")


########### The below function is for showing negative influence tags to Sayan Sir
def give_negative_tags(reverse_graph,r): # where r is the budget of the number of positive tags
    dict1={} # key : influence tags , value : number of occurence of influence tags
    for follower in reverse_graph:
        for influencial_user in  reverse_graph[follower]:
            for i in reverse_graph[follower][influencial_user]:
                event_tag=i[0]
                if(i[2]=='neg'):
                    if(event_tag in dict1):
                        number=dict1[event_tag]
                        dict1[event_tag]=number+1
                    else:
                        
                        dict1[event_tag]=1
    sorted_dict=sort_dictionary_by_value(dict1)                    
    dict1={}
    index=1
    for influence_tag in sorted_dict:
        dict1[influence_tag]=sorted_dict[influence_tag]
        index=index+1
        if(index == r):
            break
        
    return dict1
#dict_for_influence_tags=give_negative_tags(reverse_graph,20)


def give_positive_tags(reverse_graph,r): # where r is the budget of the number of positive tags
    dict1={} # key : influence tags , value : number of occurence of influence tags
    for follower in reverse_graph:
        for influencial_user in  reverse_graph[follower]:
            for i in reverse_graph[follower][influencial_user]:
                event_tag=i[0]
                if(i[2]=='pos'):
                    if(event_tag in dict1):
                        number=dict1[event_tag]
                        dict1[event_tag]=number+1
                    else:
                        
                        dict1[event_tag]=1
    sorted_dict=sort_dictionary_by_value(dict1)                    
    dict1={}
    index=1
    for influence_tag in sorted_dict:
        dict1[influence_tag]=sorted_dict[influence_tag]
        index=index+1
        if(index == r):
            break
        
    return dict1

def give_all_tags(dict_for_target_users,reverse_graph):
    dict1={} # key : influence tags , value : number of occurence of influence tags
    for follower in dict_for_target_users:
        for influencial_user in  reverse_graph[follower]:
            for i in reverse_graph[follower][influencial_user]:
                event_tag=i[0]
                
                if(event_tag in dict1):
                    number=dict1[event_tag]
                    dict1[event_tag]=number+1
                else:
                    
                    dict1[event_tag]=1
    sorted_dict=sort_dictionary_by_value(dict1)                    
    dict1={}
    index=1
    for influence_tag in sorted_dict:
        dict1[influence_tag]=sorted_dict[influence_tag]
        index=index+1
        
        
        
    return dict1
#dict_for_influence_tags=give_positive_tags(reverse_graph,20)