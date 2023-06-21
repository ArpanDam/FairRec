# Aim is to find r influence tags for which the activation of target by seed users is maximum
import pickle
import operator
import time as T
import dictionary_from_bfs
from queue import Queue
import numpy as np
import generate_original_graph_for_input

start=T.time()

### Global variabales ###

#tags_in_path={} #key:tag #value:number of times that tag occurs in the path
#path_global=[] #list of edges, each edge is identified by a 3-tuple: (from,to,tag) 
#path_tagset={}#key:path #val:tag_set of the path, it is basically the table of slide 19

#########################

			

def generate_edge_prob2(edge_prob):
    edge_prob2={}
    for uid in edge_prob:
   		for mid in edge_prob[uid]:
   			for (t,p) in edge_prob[uid][mid]:
   				edge_prob2[(uid,mid,t)]=p
   	#pickle.dump(edge_prob2,open("edge_prob2","wb"))
    return edge_prob2    


def pr(path): #prints the path
    for i in range(len(path)-1):
        pass
		#print(path[i], end='-->')
	#print(path[-1])
	#print()



def func(node_list,i,r,edge_prob,seed,path_tagset,path_global,tags_in_path): #traverses all paths between a seed node and a target node

	if( len(tags_in_path)>r) or (node_list[i] in seed and len(path_global)):	
		return 
	if i==len(node_list)-1: #if we have reached the target nde
		temp=tuple(path_global)
		path_tagset[temp]=set(tags_in_path.keys())
		pr(path_global) #prints the path
		return 
	if node_list[i+1]== 112792742 and node_list[i+1] not in edge_prob[node_list[i]]:
		#print("r=",r)
		#print("node_list[i]=",node_list[i]) #185589766
		#print("node_list",node_list)
		exit()
	for (t,p) in edge_prob[node_list[i]][node_list[i+1]]:
		if t in tags_in_path:
			tags_in_path[t]+=1
		else:
			tags_in_path[t]=1
		path_global.append((node_list[i],node_list[i+1],t))
		func(node_list,i+1,r,edge_prob,seed,path_tagset,path_global,tags_in_path)
		if(tags_in_path[t]==1):
			tags_in_path.pop(t) #revert				
		else:
			tags_in_path[t]-=1
		path_global.pop() #revert		









def compute_tags_pathbatch(r,edge_prob,seed,all_shortest_paths,path_tagset,path_global,tags_in_path):
	#find all shortest paths from target to seed users and fill path_tagset
	for B in all_shortest_paths: #B is a target user
		#print("THIS==",all_shortest_paths[B])
		for A in all_shortest_paths[B]: # A is an influential user
			if A in seed: # if A is a seed user
				for path in all_shortest_paths[B][A]:
					temp=[]
					for node in path:
						temp.append(node)
					temp.reverse()
					# if(temp==[185589766, 112792742, 2794624]):
					# 	print("path=",path)
					# 	print("temp=",temp)
					# 	print("A=",A)
					# 	print("B=",B)
					# 	print("HELLO")
					# 	exit()
					func(temp,0,r,edge_prob,seed,path_tagset,path_global,tags_in_path)				

	#print(len(path_tagset))

	tags_pathbatch={} #key:tuple of tags(cannot use set instead of tuple beacuse set is unhashable) , #val: path-batch, i.e., list of paths corresponding to those tags
	for path in path_tagset:
		tup=tuple(path_tagset[path])
		if tup in tags_pathbatch:
			tags_pathbatch[tup].append(path)
		else:
			tags_pathbatch[tup]=[path]

	for tags in tags_pathbatch:
		#check all subsets of tags, if any of the subsets is present as a key in the dictionary, add its val in tags_pathbatch[tags]		
		n=len(tags)
		for i in range((1<<n)-1): #consider every subset except the set containing all elements
			s=set()
			for j in range(n):
				if((1<<j)&i):
					s.add(tags[j])
			tup=tuple(s)
			if tup in tags_pathbatch:
				for path in tags_pathbatch[tup]:
					tags_pathbatch[tags].append(path)
	return tags_pathbatch


def return_gain_r_tags(r,edge_prob,edge_prob2,seed,all_shortest_paths,path_tagset,path_global,tags_in_path):
    tags_pathbatch=compute_tags_pathbatch(r,edge_prob,seed,all_shortest_paths,path_tagset,path_global,tags_in_path)
    #print("len(tags_pathbatch)=",len(tags_pathbatch))
    return tags_pathbatch
    







def generate_edge_prob(graph): # edge prob is the reverse of the intermediate graph
    dict1={} # reverse of graph dictionary , key = follower value=dic with key influencial member
        # and value a list of event, prob and sign
    for follower in graph:
        for influencial_member in graph[follower]:
            temp={}
            list1=[] # stores the list of tuples containing event prob and sign
            for i in  graph[follower][influencial_member]:
                list1.append((i[0],i[1]))
                    
                temp[follower]=list1
                if(influencial_member not in dict1):
                    dict1[influencial_member]=temp
                else:
                    dict1[influencial_member][follower]=list1
                        #print("")
                    
   
    return (dict1)
    
'''edge_prob_old=pickle.load(open("edge_probability_career","rb"))
sign=pickle.load(open("../Example of career group tags and its dictionary of positive negative tags_0.8/influence_tags_sign_5","rb"))       
group_tags=np.load("../Example of career group tags and its dictionary of positive negative tags_0.8/group_tags_5.npy", mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')

graph=generate_original_graph_for_input.generate_final_graph(sign)
reverse_graph=generate_original_graph_for_input.generate_reverse_graph(graph)
dict_for_target_users=generate_original_graph_for_input.give_target_user(reverse_graph,400)'''
#list_of_intermediate_graph,list_of_dictionary_of_shortest_path=dictionary_from_bfs.generating_many_list_of_shortest_path(reverse_graph,0.1,200,dict_for_target_users)
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
    if(len(negatively_activated_node)>0):
        pass        
    if(len(positively_activated_node) > len(negatively_activated_node)):
        target_node=[]
        target_node.append(target_node1)
        return ("pos") # we only need the positive node
    else:
        target_node1=[]
        return("neg")    # Return no node since number of negatively activated node > number of positively activated node
def finding_number_of_activated_target_user_from_tags(S,tags,intermediate_graph,dictionary_for_target_users):
    # have to use tags 
    Number_of_positively_activated_nodes=0
    
    # need to know the target users
    target_user=set()
    for member in dictionary_for_target_users:
        target_user.add(member)
    
    
    
    new_intermediate_graph={} # this graph will contain only the tags
    
    for follower in intermediate_graph:
        temp={}
        for influencial_member in intermediate_graph[follower]:
            list1=[]
            for i in   intermediate_graph[follower][influencial_member]:
                if(i[0] in tags):
                    list1.append((i[0],i[1],i[2]))
            if(len(list1)>0):
                temp[influencial_member]=list1
        if(bool(temp)):            
            new_intermediate_graph[follower]=temp
    
    # find the reverse of the    new_intermediate_graph
     
    dict1={}
    for follower in new_intermediate_graph:
        for influencial_member in new_intermediate_graph[follower]:
            temp={}
            list1=[]
            for i in new_intermediate_graph[follower][influencial_member]:
                list1.append(i)
                temp[follower]=list1
                if(influencial_member not in dict1):
                    dict1[influencial_member]=temp
                else:
                    dict1[influencial_member][follower]=list1


        
    intermediate_graph_temp=dict1           # intermediate graph is the new intermdiate graph
    
    spread_influenced_seed=[]
    # Loop over the Monte-Carlo Simulations
    
    for _ in range(1):
        dict1={} # key follower value is a list telling if negatively or positvely activated
        # Simulate propagation process      
        new_active, A = S[:], S[:]
        
        
        while new_active:
            
            new_ones=set()    
            for influencial_member in new_active:
                if(influencial_member not in intermediate_graph_temp):
                    continue
                for follower in intermediate_graph_temp[influencial_member]:
                    new_ones.add(follower)
                    if(follower not in target_user):
                        continue
                    postive_or_negative=positive_node_and_path_finder(intermediate_graph_temp, influencial_member, follower,intermediate_graph_temp)    
                    if(follower in dict1):
                        list1=[]
                        list1=dict1[follower]
                        list1.append(postive_or_negative)
                    else:
                        list1=[]
                        list1.append(postive_or_negative)
                        dict1[follower]=list1
                        
            # find new new_active 
            new_active = list((new_ones) - set(A))
            
            A += new_active
        number_of_positively_activated_target_user=0   
        for follower in dict1:
            number_of_pos=0
            number_of_neg=0
            for sign in dict1[follower]:
                if sign =='pos':
                    number_of_pos=number_of_pos+1
                else:
                    number_of_neg=number_of_neg+1
            if(number_of_pos > number_of_neg):
                number_of_positively_activated_target_user=number_of_positively_activated_target_user+1    
        
            
        #spread_influenced_seed.append(number_of_positively_activated_target_user)
    return  number_of_positively_activated_target_user
def output_dictionary_of_tag_batch_with_gain(seed,number_of_sampled_graph,list_of_intermediate_graph,list_of_dictionary_of_shortest_path,r,dictionary_for_target_users):
    
    gain={}
    flag=0 
    best_gain=0
    list_of_tags_gain=[]
    final_tag_gain={}   # this is the dictionary to be returned
    number_of_sampled_graph=len(list_of_intermediate_graph)
    best_influence_tags=set()
    while(len(best_influence_tags)<=r):
        final_tag_gain={}
        for i in range(number_of_sampled_graph): # 2 is the number of sampled graph
            #print (i)
            tags_in_path={} #key:tag #value:number of times that tag occurs in the path
            path_global=[] #list of edges, each edge is identified by a 3-tuple: (from,to,tag) 
            path_tagset={}#key:path #val:tag_set of the path, it is basically the table of slide 19
            all_shortest_paths=list_of_dictionary_of_shortest_path[i]
            intermediate_graph=list_of_intermediate_graph[i]
            edge_prob=generate_edge_prob(intermediate_graph)
            edge_prob2=generate_edge_prob2(edge_prob)
            #tags_pathbatch=compute_tags_pathbatch(5,edge_prob,seed,all_shortest_paths)
            tags_pathbatch=return_gain_r_tags(r,edge_prob,edge_prob2,seed,all_shortest_paths,path_tagset,path_global,tags_in_path)
                
            number_of_positively_activated_member=0
            #tags_set=set()
            # Finding number of number_of_positively_activated_member in each tags_pathbatch
            for tags in tags_pathbatch:
                number_of_new_tag=0
                number_of_old_tag=len(best_influence_tags)
                set_new_tag=set()
                set_old_tag=set()
                set1=set()
                for individual_tag in best_influence_tags:
                    try:
                        if(len(individual_tag[0])>1):
                            set_old_tag.add(individual_tag[0])
                            set1.add(individual_tag[0])
                        else:
                            set_old_tag.add(individual_tag)
                            set1.add(individual_tag) 
                    except:
                        set_old_tag.add(individual_tag)
                        set1.add(individual_tag)    
                for individual_tag in tags:
                     set_new_tag.add(individual_tag)
                     set1.add(individual_tag)
                # Find if there is any new tag by substracting one set from another
                if ((len(set_new_tag - set_old_tag)>0) and len(set_new_tag - set_old_tag)<=r-len(best_influence_tags)):
                    number_of_positively_activated_member_union=finding_number_of_activated_target_user_from_tags(seed,set1,intermediate_graph,dictionary_for_target_users)
                    number_of_positively_activated_member_best_tags=finding_number_of_activated_target_user_from_tags(seed,set_old_tag,intermediate_graph,dictionary_for_target_users)
                    number_of_positively_activated_member=number_of_positively_activated_member_union-number_of_positively_activated_member_best_tags
                    '''list1=[]
                    for ind in set1:
                        list1.append(ind)'''
                    set2 = tuple(set_new_tag - set_old_tag)    
                    if(set2 not in final_tag_gain):
                        final_tag_gain[set2]=  number_of_positively_activated_member
                    else:
                        earlier_number_of_influenced_member=final_tag_gain[set2]
                        final_tag_gain[set2]= earlier_number_of_influenced_member+number_of_positively_activated_member
                    number_of_positively_activated_member=0    
        print("")
        
        for key in final_tag_gain:
            number_of_tags=len(key)
            tempo=final_tag_gain[key]/number_of_tags
            final_tag_gain[key]=tempo
        for key in final_tag_gain:
            tempo=final_tag_gain[key]/number_of_sampled_graph
            final_tag_gain[key]=tempo
            
        final_tag_gain_sorted=dict(sorted(final_tag_gain.items(), key=lambda item: item[1],reverse=True))
        
        length_of_tags=len(best_influence_tags)
        
        for key in   final_tag_gain_sorted:
            if(final_tag_gain_sorted[key]>0):
                for individual_key in key:
                    #if(individual_key != 'P' )
                    
                    best_influence_tags.add(individual_key)
                    best_gain=best_gain+final_tag_gain_sorted[key]
            else:
                flag=1
                break
            if(len(best_influence_tags)>length_of_tags) or (flag ==1):    
                break    
        print(best_influence_tags)
        if(len(best_influence_tags)==r) or (flag==1):
            break
    '''filehandler = open("final_tag_gain","wb")
    pickle.dump(final_tag_gain,filehandler)
    filehandler.close()'''   
    return  best_influence_tags   
        
        
    # for each key in tags_pathbatch find the 
    #return tags_pathbatch
    '''for individual_tag_gain in tags_gain:
        if(individual_tag_gain in final_tag_gain):
            final_tag_gain[individual_tag_gain]= final_tag_gain[individual_tag_gain] + tags_gain[individual_tag_gain]
        else:
            final_tag_gain[individual_tag_gain]=tags_gain[individual_tag_gain]    
    list_of_tags_gain.append(tags_gain)
    #print("") 
    
return final_tag_gain'''
    
    
    



