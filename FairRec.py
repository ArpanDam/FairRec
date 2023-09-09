# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 14:37:36 2023

@author: HP
"""

import pickle
import numpy as np
import generate_original_graph_for_input
import random
import RIS_v2
from collections import Counter 
#import Evaluation
import Evaluation2
import dictionary_from_bfs
import tag_finder
import third_block
import best_r_tags_shortest_path_new_defintion_copy_checking_last_gain
import sys
'''edge_probability_career=pickle.load(open("edge_probability_career","rb")) # portion of the graph

file_name_sign="influence_tags_sign_"+str(10)
sign_dictionary=pickle.load(open("../Example of career group tags and its dictionary of positive negative tags_0.85/"+file_name_sign,"rb"))  # portion of the graph
graph=generate_original_graph_for_input.generate_final_graph(sign_dictionary,edge_probability_career)
graph_copy=graph'''
if(len(sys.argv) == 3):
    
    r=int(sys.argv[2]) # Number of influence tags
    k=int(sys.argv[1]) # Number of Influencial users

if(len(sys.argv) != 3):
    
    k=5 # Number of influencial users
    r=3# Number of Influence tags
graph=pickle.load(open("graph","rb"))
#shprtest_path_tag=pickle.load(open("shprtest_path_tag","rb"))
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

number_of_positive_tags=set()
for inf in graph:
    for follower in graph[inf]:
        for i in graph[inf][follower]:
            event_sign=i[2]
            if(event_sign=='neg'):
                print("Negative")
            number_of_positive_tags.add(i[0])    
node_topic=pickle.load(open("node_topic","rb")) # portion of the graph


dict1={}

'''for i in range(1,16):
    dict1[i]=0

for key in node_topic:
    if(node_topic[key])==1:
        dict1[1]=dict1[1]+1
    if(node_topic[key])==2:
        dict1[2]=dict1[2]+1
    if(node_topic[key])==3:
        dict1[3]=dict1[3]+1
    if(node_topic[key])==4:
        dict1[4]=dict1[4]+1
    if(node_topic[key])==5:
        dict1[5]=dict1[5]+1
    if(node_topic[key])==6:
        dict1[6]=dict1[6]+1
    if(node_topic[key])==7:
        dict1[7]=dict1[7]+1
    if(node_topic[key])==8:
        dict1[8]=dict1[8]+1
    if(node_topic[key])==9:
        dict1[9]=dict1[9]+1
    if(node_topic[key])==10:
        dict1[10]=dict1[10]+1 
    if(node_topic[key])==11:
        dict1[11]=dict1[11]+1
    if(node_topic[key])==12:
        dict1[12]=dict1[12]+1
    if(node_topic[key])==13:
        dict1[13]=dict1[13]+1
    if(node_topic[key])==14:
        dict1[4]=dict1[14]+1
    if(node_topic[key])==15:
        dict1[15]=dict1[15]+1'''     

for i in range(1,6):
    dict1[i]=0

for key in node_topic:
    if(node_topic[key])==1:
        dict1[1]=dict1[1]+1
    if(node_topic[key])==2:
        dict1[2]=dict1[2]+1
    if(node_topic[key])==3:
        dict1[3]=dict1[3]+1
    if(node_topic[key])==4:
        dict1[4]=dict1[4]+1
    if(node_topic[key])==5:
        dict1[5]=dict1[5]+1
      
    
dict1={k: v for k, v in sorted(dict1.items(), key=lambda item: item[1])}

lowest_attribute=list(dict1.keys())[0]
print("")
Number_of_sample_graphs=50
    
reverse_graph=generate_original_graph_for_input.generate_reverse_graph(graph)   
reverse_graph_copy=reverse_graph
dict_topic_rrs,all_rrs_set=RIS_v2.genrating_many_sampled_graph(reverse_graph,Number_of_sample_graphs,dict1,node_topic)
#dict_topic_rrs store the rrs for each attritute
# Find the attribute min
# find the best seed for lowest attribute
def seed_finder_lowest_attribute(dict_topic_rrs,all_rrs_set,lowest_attribute):
    # k is the budget of the influencial member
    SEED=[]
    for _ in range(1):
        
        # Find node that occurs most often in R and add to seed set
        flat_list = [item for sublist in dict_topic_rrs[lowest_attribute] for item in sublist]
        seed = Counter(flat_list).most_common()[0][0]
        SEED.append(seed)
        
        # Remove RRSs containing last chosen seed 
        #RRS_set = [rrs for rrs in all_rrs_set if seed not in rrs]   
    #all_rrs_set=RRS_set
    return SEED    
seed=seed_finder_lowest_attribute(dict_topic_rrs,all_rrs_set,lowest_attribute)



# Find r best tag
# find the members of lowest attribute the find the nodes of the lowest attribute 

node_with_lowest_attribute=set()
for node in node_topic:
    if node_topic[node]==lowest_attribute:
        node_with_lowest_attribute.add(node)
 
# got node with the lowest_attribute
# now find the best tags 
list_of_intermediate_graph,list_of_dictionary_of_shortest_path=dictionary_from_bfs.generating_many_list_of_shortest_path_part1(reverse_graph,Number_of_sample_graphs,node_with_lowest_attribute,seed)

set_key1=best_r_tags_shortest_path_new_defintion_copy_checking_last_gain.output_dictionary_of_tag_batch_with_gain(seed,Number_of_sample_graphs,list_of_intermediate_graph,list_of_dictionary_of_shortest_path,3,node_with_lowest_attribute)

best_tag=set_key1
print(" Best tags are ", best_tag)
print("")
def rr_set_covered_by_attribute_of_seed(seed): # output the RR set of each attribute covered 
    dict_rrs_seed={}                                                             # by seed 
    number_of_rr_set=0
    for ind_attribute in dict1:
        number_of_rr_set=0
        for ind_seed in seed:
            # find attribute of ind_seed
            #attribute=node_topic[ind_seed]
            #attribute=node_topic[ind_seed]
            RRS_set = [rrs for rrs in dict_topic_rrs[ind_attribute] if ind_seed in rrs]
            number_of_rr_set=number_of_rr_set+len(RRS_set)
        dict_rrs_seed[ind_attribute]=number_of_rr_set    
    return dict_rrs_seed   

def rr_set_covered_by_seed(seed):
    number_of_rr_set=0
    for ind_seed in seed:
        # find attribute of ind_seed
        #attribute=node_topic[ind_seed]
        RRS_set = [rrs for rrs in all_rrs_set if ind_seed in rrs]
        number_of_rr_set=number_of_rr_set+len(RRS_set)
    return number_of_rr_set      
def find_lowest_attribute(seed,all_rrs_set):
   
    #rr_set_covered_by_seed= [rrs for rrs in all_rrs_set if seed in rrs]
    number_of_rr_set_covered_by_seed=rr_set_covered_by_seed(seed)
    dict_rrs_seed=rr_set_covered_by_attribute_of_seed(seed)
    expected_coverafe_each_attribute={}
    for attribute in dict_rrs_seed:
        expected_coverafe_each_attribute[attribute]=(dict_rrs_seed[attribute]/number_of_rr_set_covered_by_seed)
    expected_coverafe_each_attribute={k: v for k, v in sorted(expected_coverafe_each_attribute.items(), key=lambda item: item[1])} 
    return list(expected_coverafe_each_attribute.keys())[0]

def seed_finder_lowest_attribute_after_1st_iteration(dict_topic_rrs,all_rrs_set,lowest_attribute):
    # k is the budget of the influencial member
    SEED=[]
    for _ in range(1):
        
        # Find node that occurs most often in R and add to seed set
        flat_list = [item for sublist in dict_topic_rrs[lowest_attribute] for item in sublist]
        seed = Counter(flat_list).most_common()[0][0]
        SEED.append(seed)
    
        
        # Remove RRSs containing last chosen seed 
        #RRS_set = [rrs for rrs in all_rrs_set if seed not in rrs]   
    #all_rrs_set=RRS_set
    return SEED

def remove_rrs_set(all_rrs_set,dict_topic_rrs,seed):
    for ind_seed in seed:
       
        RRS_set = [rrs for rrs in all_rrs_set if ind_seed not in rrs]
        all_rrs_set=RRS_set
        for key in dict_topic_rrs:
            RRS_set_topic=[]
            RRS_set_topic=[rrs for rrs in dict_topic_rrs[key] if ind_seed not in rrs]
            dict_topic_rrs[key]=RRS_set_topic
    return all_rrs_set,dict_topic_rrs

k_range=[k]
for k in k_range:
    while(len(seed)<k):    
        lowest_attribute=find_lowest_attribute(seed,all_rrs_set)
        # Find the node with the lowest attribute
        #node_with_lowest_attribute=set()
        for node in node_topic:
            if node_topic[node]==lowest_attribute:
                node_with_lowest_attribute.add(node)
        # now find the best tags   
        list_of_intermediate_graph,list_of_dictionary_of_shortest_path=dictionary_from_bfs.generating_many_list_of_shortest_path_part1(reverse_graph,Number_of_sample_graphs,node_with_lowest_attribute,seed)

        set_key1=best_r_tags_shortest_path_new_defintion_copy_checking_last_gain.output_dictionary_of_tag_batch_with_gain(seed,Number_of_sample_graphs,list_of_intermediate_graph,list_of_dictionary_of_shortest_path,r,node_with_lowest_attribute)

        best_tag=set_key1
        print(" Best tags are ", best_tag)
        # now remove the RR set of seed
        all_rrs_set,dict_topic_rrs=remove_rrs_set(all_rrs_set,dict_topic_rrs,seed)
        seed2=seed_finder_lowest_attribute_after_1st_iteration(dict_topic_rrs,all_rrs_set,lowest_attribute)
        for ind_seed in seed2:
            seed.append(ind_seed)
        print("")
    
    #[10297783, 3437293, 49753142, 62567262, 55880202, 9452623] 5 seed
    #[10297783, 3437293, 49753142, 62567262, 55880202, 9452623, 11518904, 11668788, 2621867, 10142001, 75088462]
    #[10297783, 3437293, 49753142, 62567262, 55880202, 9452623, 11518904, 11668788, 2621867, 10142001, 75088462, 5015841, 2794624, 184195777, 13831197, 98382572]
    
    #find_lowest_attribute(seed,all_rrs_set)
    #seed=[80674, 2775988, 2794624, 10205905, 37955902]
    #seed=[80674, 2775988, 2794624, 2884706, 7130244, 10205905, 12140271, 28627972, 29208582, 37955902]
    #seed=[80674, 2775988, 2794624, 2884706, 3208280, 7092500, 7130244, 9078175, 10205905, 11900080, 12140271, 29208582, 37955902, 39398622, 186430178]
    #seed=[80674, 2775988, 2794624, 2884706, 6402821, 7092500, 7130244, 9078175, 10021896, 10205905, 11900080, 12140271, 28627972, 29208582, 37955902, 39398622, 87246662, 127409112, 186430178, 190845616]
    #seed=[80674, 2775988, 2794624, 2884706, 6402821, 7092500, 7130244, 8840408, 9078175, 10021896, 10205905, 10524205, 11168170, 11900080, 12140271, 28627972, 29208582, 34084072, 37955902, 39398622, 87246662, 127409112, 186038049, 186430178, 190845616]
    #seed=[80674, 2775988, 2794624, 2884706, 6402821, 7018478, 7092500, 7130244, 7964051, 8840408, 9078175, 10021896, 10205905, 10524205, 11168170, 11900080, 12140271, 13149935, 28627972, 29208582, 34084072, 37955902, 39398622, 87246662, 117906492, 127409112, 182476136, 186038049, 186430178, 190845616]
    print(seed)
    print(len(seed))
    #seed=[80674, 2775988, 2794624, 9078175, 10205905, 11900080, 37955902]
    #seed=[321158, 2775988, 2794624, 9078175, 10205905, 11900080, 37955902]
    #seed=[80674, 2775988, 2794624, 7130244, 10205905, 28627972, 37955902]
    #best_tag= {'Making Friends to Travel With', 'Be the Change You Wish to See in the World', 'fun times- good meals- and new friends!', 'How to start a business', 'Exercise and Have Fun at the same time', 'Learn How to Be a Success in Network Marketing', 'How To Use Social Media To Promote Your Business'}
    number_of_influenced,influenced_member=Evaluation2.number_of_influenced_member(seed,reverse_graph,best_tag)
    ''' Evaluation_number_of_target_nodes_positively_activated_part2.number_of_influenced_member(top_influencial_member,reverse_graph,dict_for_influence_tags)
    list_of_intermediate_graph,list_of_dictionary_of_shortest_path=dictionary_from_bfs.generating_many_list_of_shortest_path(reverse_graph,Number_of_sample_graphs,influenced_member,seed)
    tag=tag_finder.tag_finder(list_of_dictionary_of_shortest_path)
    print(tag)
    print(len(tag))'''
    def finding_percentage_of_influenced_member(influenced_member):
        dict_percentage_influenced_member={}
        for member in influenced_member:
            topic=node_topic[member]
            if topic not in dict_percentage_influenced_member:
                dict_percentage_influenced_member[topic]=1
            else:
                dict_percentage_influenced_member[topic]=dict_percentage_influenced_member[topic]+1
        total_member=0
        for key in dict_percentage_influenced_member:
            total_member= dict_percentage_influenced_member[key]+total_member 
        for key in dict_percentage_influenced_member:
            dict_percentage_influenced_member[key]=dict_percentage_influenced_member[key] /total_member     
        return dict_percentage_influenced_member            
    print("")
    
    dict_percentage_influenced_member=finding_percentage_of_influenced_member(influenced_member)
    
    def finding_percentage_of_member_in_whole_graph(node_topic):
        total_member=len(node_topic)
        dict_percentage_member={}
        for i in range(1,6):
            a=[k for k, v in node_topic.items() if v == i]
            dict_percentage_member[i]=len(a)/total_member
        
        return dict_percentage_member
    print("")
    
    
    dict_percentage_member=finding_percentage_of_member_in_whole_graph(node_topic)
    
    
    def find_difference_between_percentage(dict_percentage_influenced_member,dict_percentage_member):
        diff=0
        for key in dict_percentage_influenced_member:
            diff=diff+abs(dict_percentage_member[key]-dict_percentage_influenced_member[key])
        return diff
    print("Total percentage influenced is ",number_of_influenced)
    #print("influenced member percentage =",dict_percentage_influenced_member)
    #print("All member percentage =",dict_percentage_member)
    
    print("L1 norm is ",find_difference_between_percentage(dict_percentage_influenced_member,dict_percentage_member))
    #break
#[80674, 2775988, 2794624, 9078175, 10205905, 11900080, 37955902]

print("Best influential users are",seed)
print("Best influence tags are", best_tag)
# Begining of 3rd block

