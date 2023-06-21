# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 09:12:39 2022

@author: HP
"""

import pickle
import pandas as pd
import random
from random import uniform, seed
import numpy as np
#from igraph import *
import pickle
#import matplotlib.pyplot as plt
#from scipy import stats
from numpy import save
from numpy import load
from queue import Queue
#from igraph import *
import matplotlib.pyplot as plt
import generate_original_graph_for_input
#import find_seed_from_influence_tags_having_many_graph


#edge_probability_career = pickle.load(open("edge_probability_career","rb"))
#sign_dictionary=pickle.load(open("../Example of career group tags and its dictionary of positive negative tags_0.8/influence_tags_sign_5","rb"))        
#group_tags=np.load("../Example of career group tags and its dictionary of positive negative tags_0.8/group_tags_5.npy", mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
# 15 group tags 


#edge_prob = pickle.load(open("edge_probability_career","rb"))#key: influencer id, value: dict with key as mid, value as list of tuples(t,p), where t is the event tag and p is the probability

#graph=generate_original_graph_for_input.generate_final_graph(sign_dictionary)
#reverse_graph=generate_original_graph_for_input.generate_reverse_graph(graph)
#dict_for_target_users=generate_original_graph_for_input.give_target_user(reverse_graph,400) # 400 is the number of target user
#dict_for_influence_tags=generate_original_graph_for_input.give_best_postive_tags(dict_for_target_users,reverse_graph,8)   # 5 is the number of tags
#dict_for_influence_tags_top_11_tags={'Learn How to Be a Success in Network Marketing': 727, 'Public Speaking as a Means to Market your Business': 436, '"Cloud Computing" How to use it for your business': 311, 'Seminars on First Time Home Buyers': 150, 'Making Friends to Travel With': 142, 'How to start a business': 141, 'How to make money in network marketing': 130, 'How To Use Social Media To Promote Your Business': 110, 'Market Your Small Business Locally On The Internet': 54, 'The Human Energy Field: A Practical Introduction': 42, 'Learn Mind Tools for Expanding Consciousness': 42}
#dict_for_influence_tags_top_11_tags={'Learn How to Be a Success in Network Marketing': 730, 'Public Speaking as a Means to Market your Business': 446, '"Cloud Computing" How to use it for your business': 310, 'How to start a business': 143, 'How to make money in network marketing': 131, 'How To Use Social Media To Promote Your Business': 118, 'How To Build Business Credit': 37}
 
#dict_for_remaining_tags={'How to start a business': 141, 'How to make money in network marketing': 130, 'How To Use Social Media To Promote Your Business': 110, 'Market Your Small Business Locally On The Internet': 54, 'The Human Energy Field: A Practical Introduction': 42, 'Learn Mind Tools for Expanding Consciousness': 42}
#dict_for_all_tags={'Learn How to Be a Success in Network Marketing': 596, 'Public Speaking as a Means to Market your Business': 440, '"Cloud Computing" How to use it for your business': 285, 'Exercise and Have Fun at the same time': 280, 'Seminars on First Time Home Buyers': 155, 'Making Friends to Travel With': 149, 'How to start a business': 138, 'How to make money in network marketing': 126, "Let's get a Furmeet setup and make a home here": 107, 'Making a Difference in the World': 88, 'The Secret (DVD) Making It Work For You': 79, 'Meetings other couples who share your interest too': 78, 'Dining Out, BBQs, Food Fairs, Happy Hour and More': 76, 'How To Use Social Media To Promote Your Business': 68, 'Fine Dining Happy Hours and Great New Friends': 57, 'Market Your Small Business Locally On The Internet': 55, 'Be the Change You Wish to See in the World': 48, 'fun times- good meals- and new friends!': 45, "Meeting other SAHM's": 42, 'How To Build Business Credit': 32, 'The Human Energy Field: A Practical Introduction': 27, 'social events for everyone 40+': 26, 'Fun for Friends and Their Friends': 25, 'Meeting New and Exciting People': 22, 'Make Some New Friends': 22, 'Learn Mind Tools for Expanding Consciousness': 21, 'Healthy Cooking For You and Your Family': 20, 'Dinner and a Movie': 19, 'Building Network Marketing Skills and Strategies': 19, 'Social Networking Make New Friends': 18, 'What to look for when purchasing a new home': 18, 'Develop Clean and Clear Algorithms': 18, 'Afterwork Happy Hours and Dinner': 16, 'Making friends in our Community': 16, 'Share Experiences Knowledge of Product Development': 16, 'Date Night': 15, 'Thinking About Divorce: Sort out the Process': 15, 'Meeting other parents': 14, 'Giving Back to the Community - Engaging Our Youth': 14, 'How Do I Begin My Home Search': 14, 'House wives that need a night out.': 11, 'Business Opportunity Meeting': 11, 'Using your Public speaking to increase your income': 11, 'How To Dance and Dance For Fun': 11, 'Where or where to go on a honeymoon?': 11, 'Why We Need Highly Sensitive People': 11, 'Activities': 10, 'Anyone craving to play a game of real MAHJONG?': 10, 'Share New Ideas that Work': 10, 'How to Maintain your Natural Hair in Winter Time': 9, 'Meet New Nannies in the Same Age Group': 9, "Community Service for Age 50's+: Various Projects": 9, 'Learn How to Stop Foreclosure': 8, 'Internet Marketing Strategies For Business Owners': 8, 'Meeting New People/Making Friends': 8, 'How does Medical Marijuana Work': 8, 'Make Money Online': 7, 'Time for Lunch': 7, 'Getting in Shape': 7, 'Meeting old friends and making new friends': 7, 'Assist people to start businesses': 7, 'Improve your Local Business Area and Grow Profits': 7, 'Mid-Life Women- Do You Accept This For What It Is': 7, 'Social Media Marketing For Small Business': 7, 'Food and Drink': 6, 'Group Discussions of Current Events': 6, 'Have a Glass of Wine and Talk about a Good Book': 6, 'Net Tuesday': 6, 'Networking For the Self Employed': 6, 'Business Strategy and Professional Development': 6, 'play bunco and meet new friends': 5, 'Working from Home Effectively': 5, 'Reading and discussion': 5, 'Interested In Learning About Home Based Business': 5, 'Learning to Speak and Be Heard': 5, 'Getting Published': 5, 'Resonsive Web Design and Umbraco': 5, 'Internet of Things (IOT) Device Management': 5, 'Low Impact Exercise': 4, 'Dining Out With New Friends': 4, 'Learn New Skills and Artistic Expression': 4, 'Best Way to Sleep at Night': 4, 'Weekend Actvities': 4, 'Have you had a Spiritual Experience': 4, 'Social Media for Small Business': 4, 'Big Data for social media': 4, 'Business to Business Projects and Networking': 4, 'Movie discussion meet up': 4, 'Girls Only Social Site': 3, 'Self-Publishing:  Good or Bad': 3, 'For 50+ Women Who Want to Meet New Friends': 3, 'Nannies Looking to Meet Like Minded People & Fun': 3, 'Current Events': 3, "Making Friends when you're Over 50": 3, 'Accelerated Learning: Learning How To Learn': 3, 'toastmasters is the proven way to public speaking': 3, 'Support for Parents of Special Needs Kids': 3, 'Portrait painting and drawing from live model': 3, 'Together we are strong -Lyoness shopping community': 3, 'Real Estate: First Time and Return Home Buyers': 3, 'Women Support through Creative Process': 3, 'Business Analysis tools and techniques': 3, 'Found Art': 3, "Social Skills for Teens with Asperger's Syndrome": 3, 'Do you want to meet other bronies and be friends': 3, 'work at home parents': 3, 'Business start up': 3, 'Early Morning': 2, 'Training For Fun Runs': 2, 'Creativity Workshop for Artists of All Disciplines': 2, 'Making Money with Rental Property': 2, 'Line Dancing to All Kinds of Music': 2, 'Work At Home': 2, 'Business Strategy for growth': 2, 'Investing using the CAN SLIM Investing System': 2, 'The Open Web Application Security Project': 2, 'How to Market Your Business through Blogging': 2, 'Having Fun and Making Money': 2, 'Meeting other Women in Business': 2, 'Growth': 2, 'Social Events for Women': 2, 'Selling your Home for Top Dollar': 2, 'Board game Go': 2, 'Meet interesting people': 2, 'Waiting to Adopt': 2, 'Creating Positive Change': 2, 'Try Something New': 2, 'Coffee and Chat': 1, 'Work Hard!  Play Hard!': 1, 'Event Planning': 1, 'international business events': 1, 'Networking and Professional Development': 1, "Read and discuss Rand's works": 1, 'Guys Social and Activity Group': 1, 'Real Estate Education for Buyers and Sellers': 1, 'Practicing and Learning New Languages': 1, 'Dinner and Drinks': 1, 'Start a Food Business': 1, 'Book discussion group': 1, 'Become the Bank and get the LARGE gains': 1, 'Resume Help, Job Search Workshop, Interview Help': 1, 'Live Music & Art Shows': 1, 'Executive and business coaching and mentoring': 1, 'Life Drawing drop in sessions': 1, 'Women Going through Life Changes': 1, 'Creating a Healthy Relationship to Food': 1, "First Time Home Buyer's Workshop": 1, 'Stock Market Trading through Proven Strategies': 1, 'We Are Change': 1, 'Living Your Best Life in Mind Body Spirit': 1, 'Meet up on weekends': 1, 'Full Figured People': 1, 'Happy Hours: Meet People and Socialize, Age 50+': 1, 'Children of all ages and races with special needs': 1, 'Is it hard to find bronies in your area': 1, 'Day Trips': 1, 'Change Your Mind and You Will Change Your Life': 1, 'Things to do on a BUDGET': 1, 'Meeting for lunch': 1, 'For Folks to Meet and Greet with other Small Dogs': 1, 'Better Business Referrals, Leads -vs- Referrals': 1, "Women Still Having Fun in Their 60's": 1, 'Meet single professionals over 40': 1, 'Fun Run': 1, "Let's Just Hang Out": 1, 'Fun Activities for Kids': 1, 'Print and Web Design': 1, 'start a business': 1, 'New product development model for medical device': 1, 'Testing Web Site Performance': 1, 'Doctor Who fans that want to share ideas and fun!!': 1, 'Small Business Development': 1, 'Returning to work after Stroke': 1, 'Law of Attraction & Positive Thinking': 1, 'Special Needs is Not a Bad Word': 1, 'Getting support from your neighbors': 1, 'Google Message Security': 1, '20-30': 1, 'Short Sale Help': 1, 'College Students Teacher learn Sign Languages': 1, '6.00x Fall 2012': 1, 'Winter and Summer Sports': 1, 'Table Top Role Playing Games': 1, 'Paying it forward': 1, 'Love Life Nature Health Positive People': 1, 'Team in Training': 1, 'Step by Step on how to use QuickBooks': 1, 'Where lonely people make friends Be lonely no more': 1, 'Small Groups Book Club': 1, 'Golf for Everyone': 1, 'For The People': 1, 'Discussion and Dinner': 1, 'Go game': 1}
#dict_for_all_tags={'Learn How to Be a Success in Network Marketing': 730, 'Public Speaking as a Means to Market your Business': 446, '"Cloud Computing" How to use it for your business': 310, 'Exercise and Have Fun at the same time': 216, 'Seminars on First Time Home Buyers': 143, 'How to start a business': 143, 'Making Friends to Travel With': 133, 'How to make money in network marketing': 131, 'How To Use Social Media To Promote Your Business': 118, 'Making a Difference in the World': 89, 'The Secret (DVD) Making It Work For You': 78, 'Dining Out, BBQs, Food Fairs, Happy Hour and More': 63, 'Meetings other couples who share your interest too': 60, 'Market Your Small Business Locally On The Internet': 58, 'Be the Change You Wish to See in the World': 49, 'Fine Dining Happy Hours and Great New Friends': 47, 'fun times- good meals- and new friends!': 41, "Meeting other SAHM's": 41, 'How To Build Business Credit': 37, "Let's get a Furmeet setup and make a home here": 37, 'The Human Energy Field: A Practical Introduction': 35, 'Learn Mind Tools for Expanding Consciousness': 29, 'social events for everyone 40+': 26, 'Fun for Friends and Their Friends': 25, 'Develop Clean and Clear Algorithms': 24, 'Share Experiences Knowledge of Product Development': 24, 'Social Networking Make New Friends': 23, 'Meeting New and Exciting People': 23, 'Business start up': 22, 'Building Network Marketing Skills and Strategies': 19, 'Make Some New Friends': 18, 'Business Opportunity Meeting': 18, 'Dinner and a Movie': 18, 'Share New Ideas that Work': 18, 'Thinking About Divorce: Sort out the Process': 17, 'Meeting other parents': 15, 'Internet of Things (IOT) Device Management': 15, 'Afterwork Happy Hours and Dinner': 15, 'Date Night': 13, 'Giving Back to the Community - Engaging Our Youth': 13, "Community Service for Age 50's+: Various Projects": 13, 'Healthy Cooking For You and Your Family': 13, 'How does Medical Marijuana Work': 12, 'Making friends in our Community': 12, 'What to look for when purchasing a new home': 11, 'How To Dance and Dance For Fun': 11, 'Why We Need Highly Sensitive People': 11, 'Meet New Nannies in the Same Age Group': 9, 'Networking For the Self Employed': 9, 'Internet Marketing Strategies For Business Owners': 9, 'House wives that need a night out.': 9, 'Make Money Online': 9, 'Have a Glass of Wine and Talk about a Good Book': 9, 'Improve your Local Business Area and Grow Profits': 8, 'How Do I Begin My Home Search': 8, 'Big Data for social media': 8, 'Learn New Skills and Artistic Expression': 8, 'Group Discussions of Current Events': 8, 'Net Tuesday': 7, 'Interested In Learning About Home Based Business': 7, 'Activities': 7, 'Social Media Marketing For Small Business': 7, 'Resonsive Web Design and Umbraco': 7, 'Using your Public speaking to increase your income': 7, 'Assist people to start businesses': 7, 'Mid-Life Women- Do You Accept This For What It Is': 7, 'Where or where to go on a honeymoon?': 7, 'Do you want to meet other bronies and be friends': 7, 'Getting in Shape': 6, 'Learn How to Stop Foreclosure': 6, 'Getting Published': 6, 'Living Your Best Life in Mind Body Spirit': 6, 'Short Sale Help': 6, 'Business Strategy and Professional Development': 6, 'Social Media for Small Business': 5, 'Business Analysis tools and techniques': 5, 'Try Something New': 5, 'Creating Positive Change': 5, 'Business to Business Projects and Networking': 5, 'Meeting New People/Making Friends': 5, 'How to Market Your Business through Blogging': 4, 'Time for Lunch': 4, 'play bunco and meet new friends': 4, 'Low Impact Exercise': 4, 'Learning to Speak and Be Heard': 4, 'Movie discussion meet up': 4, 'Working from Home Effectively': 4, 'Girls Only Social Site': 3, 'Nannies Looking to Meet Like Minded People & Fun': 3, 'Food and Drink': 3, 'Business Strategy for growth': 3, 'Portrait painting and drawing from live model': 3, '20-30': 3, 'Internet Marketing To Attract Local Consumers': 3, '6.00x Fall 2012': 3, 'Current Events': 3, 'Have you had a Spiritual Experience': 3, 'Learn how to Use Your Digital Camera': 3, 'work at home parents': 3, 'Reading and discussion': 3, 'How to Maintain your Natural Hair in Winter Time': 3, 'Making Money with Rental Property': 2, 'Together we are strong -Lyoness shopping community': 2, 'Investing using the CAN SLIM Investing System': 2, 'Weekend Actvities': 2, 'start a business': 2, 'Training For Fun Runs': 2, 'For 50+ Women Who Want to Meet New Friends': 2, 'Creativity Workshop for Artists of All Disciplines': 2, 'Women Support through Creative Process': 2, 'Event Planning': 2, 'Best practices for social media marketing': 2, 'Work At Home': 2, 'Life Drawing drop in sessions': 2, 'Social Events for Women': 2, 'Dining Out With New Friends': 2, 'Human Development and Evolution In These Times': 2, 'Practicing and Learning New Languages': 2, 'Leadership Development for Technical Professionals': 2, 'Growth': 2, 'Small Groups - Book Club, Discuss Issues': 2, 'Self-Publishing:  Good or Bad': 2, 'Anyone craving to play a game of real MAHJONG?': 2, 'Early Morning': 2, 'The Open Web Application Security Project': 1, 'international business events': 1, 'Real Estate Education for Buyers and Sellers': 1, 'Accelerated Learning: Learning How To Learn': 1, 'Networking and Professional Development': 1, 'Executive and business coaching and mentoring': 1, "Read and discuss Rand's works": 1, 'toastmasters is the proven way to public speaking': 1, 'Free Classes': 1, 'Doctor Who fans that want to share ideas and fun!!': 1, 'Coffee and Chat': 1, 'Getting support from your neighbors': 1, 'Google Message Security': 1, 'College Students Teacher learn Sign Languages': 1, 'Getting Things Done': 1, 'Signing Together': 1, 'Long Distance Running': 1, 'Resume Help, Job Search Workshop, Interview Help': 1, 'Live Music & Art Shows': 1, 'Returning to work after Stroke': 1, 'Law of Attraction & Positive Thinking': 1, 'Waiting to Adopt': 1, 'Special Needs is Not a Bad Word': 1, 'Support for Parents of Special Needs Kids': 1, 'Outdoor Activities for All Ages': 1, 'Become the Bank and get the LARGE gains': 1, 'Print and Web Design': 1, 'New product development model for medical device': 1, 'Meet interesting people': 1, 'Testing Web Site Performance': 1, 'Love Yourself and Attract True Love': 1, 'Creating a Healthy Relationship to Food': 1, 'Small Business Development': 1, 'Selling your Home for Top Dollar': 1, 'Stock Market Trading through Proven Strategies': 1, 'Found Art': 1, 'We Are Change': 1, 'Best Way to Sleep at Night': 1, 'Guys Social and Activity Group': 1, 'Full Figured People': 1, 'Physical and Emotional Balance with Essential Oil': 1, 'Line Dancing to All Kinds of Music': 1, 'Change Your Mind and You Will Change Your Life': 1, 'Discussion and Dinner': 1, 'Things to do on a BUDGET': 1, 'Dinner and Drinks': 1, 'Start a Food Business': 1, 'Card Making Classes': 1, 'Beer and Food Pairing': 1, 'Better Business Referrals, Leads -vs- Referrals': 1, 'Having Fun and Making Money': 1, 'Meeting other Women in Business': 1, 'Meet other beatboxers': 1, 'How to make Chinese food': 1, 'Starting an Online Business': 1, 'Life Transformation using practical tools': 1, 'offering hope & support to the homeless community': 1, 'Questions': 1, 'Work Hard!  Play Hard!': 1, 'Women Going through Life Changes': 1, "First Time Home Buyer's Workshop": 1}
#dict_for_remaining_tags={"Let's get a Furmeet setup and make a home here": 107, 'Meetings other couples who share your interest too': 78, 'Fine Dining Happy Hours and Great New Friends': 57, 'Market Your Small Business Locally On The Internet': 55, 'Fun for Friends and Their Friends': 25, 'Make Some New Friends': 22}

    
    
def generate_graph_having_only_r_tags(reverse_graph,dict_for_influence_tags): # generate graph only having r influence tags
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

def filtered_graph_generator(filtered_reverse_graph):
    dict1={}
    for follower in filtered_reverse_graph:
        for influencial_member in filtered_reverse_graph[follower]:
            temp={}
            list1=[]
            for i in filtered_reverse_graph[follower][influencial_member]:
                list1.append(i)
                temp[follower]=list1
                if(influencial_member not in dict1):
                    dict1[influencial_member]=temp
                else:
                    dict1[influencial_member][follower]=list1
    return dict1




    

def IC_dataframe_input(G,S,mc,influencial_or_random):  
    """
    Input:  G: filtered graph
            S:  Set of seed nodes
            p:  Disease propagation probability
            mc: Number of Monte-Carlo simulations
    Output: Average number of nodes influenced by the seed nodes
    """
    spread_influenced_seed=[]
    # Loop over the Monte-Carlo Simulations
    
    for _ in range(mc):
        dict1={} # key follower value is a list telling if negatively or positvely activated
        # Simulate propagation process      
        new_active, A = S[:], S[:]
        
        intermediate_graph={}
        for influencial_member in G:
            temp={}
            for follower in G[influencial_member]:
                list1=[]
                for i in  G[influencial_member][follower]: # i is an edge
                    if(random.uniform(0, 1)<i[1]):
                    # sucess
                        list1.append((i[0],i[1],i[2]))
                if(len(list1)>0):
                    # check if follower is positively 
                    #new_ones.add(follower)
                    temp[follower]=list1
            if ( bool(temp)):        
                intermediate_graph[influencial_member]=temp
        while new_active:
            
            new_ones=set()    
            for influencial_member in new_active:
                if(influencial_member not in intermediate_graph):
                    continue
                for follower in intermediate_graph[influencial_member]:
                    new_ones.add(follower)
                    
                    postive_or_negative="pos"
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
        influenced_member=set()
        for follower in dict1:
            number_of_pos=0
            
            for sign in dict1[follower]:
                if sign =='pos':
                    number_of_pos=number_of_pos+1
                    influenced_member.add(follower)
            if(number_of_pos > 0):
                number_of_positively_activated_target_user=number_of_positively_activated_target_user+1    
        if(influencial_or_random==1):
            
            spread_influenced_seed.append(number_of_positively_activated_target_user)
    return  spread_influenced_seed,influenced_member           
            
################# loop starts here ##########


#dict_for_influence_tags={'Public Speaking as a Means to Market your Business': 440, 'Exercise and Have Fun at the same time': 280, 'Seminars on First Time Home Buyers': 155, 'How to start a business': 138}



def number_of_influenced_member(S,reverse_graph,dict_for_influence_tags):

    filtered_reverse_graph=generate_graph_having_only_r_tags(reverse_graph,dict_for_influence_tags)
                    
    
    filtered_graph= filtered_graph_generator(filtered_reverse_graph)
    
    
    
    
    
    
    
    
          
   
    
    
    S=list(S)
    
    
    spread_influenced_seed,influenced_member=IC_dataframe_input(filtered_graph,S,mc=500,influencial_or_random=1)
    
    return ((sum(spread_influenced_seed)/len(spread_influenced_seed))),influenced_member

            
              
        
    


 

