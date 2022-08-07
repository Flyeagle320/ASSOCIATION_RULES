# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 11:20:57 2022

@author: Rakesh
"""

######################################problem 1##################################
#################This code is being refferred from Github from Mr Nithin Dsouza#############################
#importing packages ######################
import pandas as pd
from mlxtend.frequent_patterns import association_rules , apriori
import matplotlib.pyplot as plt

##loading dataset ##

books = pd.read_csv('D:/DATA SCIENCE ASSIGNMENT/Datasets_Association Rules/book.csv')
##checking na and null value 
books.isna().sum()
books.isnull().sum()

##applying apriori method##

books2 = apriori(books,min_support=0.075,max_len=4 ,use_colnames = True)
books2.sort_values('support',ascending = False , inplace = True)

##plotting barplot#
plt.bar(x=list(range(0,11)),height=books2.support[0:11], color=['red','blue','green''yellow','black'])
plt.xticks(list(range(0,11)),books2.itemsets[0:11], rotation=20)
plt.xlabel('itemsets')
plt.ylabel('support')
plt.show()

##association rules##

rules = association_rules(books2,metric= 'lift',min_threshold=1)
rules.head(20)
rules.sort_values('lift',ascending=False).head(10)

def to_list(i):
    return(sorted(list(i)))

new_rules = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

new_rules = new_rules.apply(sorted)
rules_sets =list(new_rules)

unique_rules_sets = [list(m) for m in set(tuple(i)for i in rules_sets)]

index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

##getting rules without redundancy##
rules_no_redundancy= rules.iloc[index_rules, :]

## sort them to list and retrieve top 10 rules##
rules_no_redundancy.sort_values('lift', ascending= False).head(10)

############################Problem2##############################################

import pandas as pd
from mlxtend.frequent_patterns import apriori , association_rules

groceries = []

with open('D:/DATA SCIENCE ASSIGNMENT/Datasets_Association Rules/groceries.csv') as f:
    groceries= f.read()
    
##splitting data into separate transactions using seperator as "\n"
groceries= groceries.split("\n")

groceries_list=[]
for i in groceries:
    groceries_list.append(i.split(","))
all_groceries_list = [i for item in groceries_list for i in item]    
    
from collections import Counter     
item_frequencies = Counter(all_groceries_list)

##after sorting##
item_frequencies= sorted(item_frequencies.items(), key= lambda x:x[1])

##storing frequencies & items in seperate variables##
frequencies= list(reversed([i[1] for i in item_frequencies]))
items= list(reversed([i[0] for i in item_frequencies]))

#barplotting #
import matplotlib.pyplot as plt
plt.bar(height=frequencies[0:11], x= list(range(0,11)), color=['red','blue','green''yellow','black'])
plt.xticks(list(range(0,11),),items[0:11])
plt.xlabel('items')
plt.ylabel('count')
plt.show()

##creating data frame for the transaction data##
groceries_series = pd.DataFrame(pd.Series(groceries_list))
groceries_series=groceries_series.iloc[: 9835 , :]

groceries_series.columns= ['transactions']

##creating dummy columns for each items in each transactions ##using column name as item name##

x= groceries_series['transactions'].str.join(sep="*").str.get_dummies(sep= "*")

frequent_itemsets = apriori(x,min_support=0.0075, max_len=4, use_colnames=True)

##most frequent itemsets based on support ##
frequent_itemsets.sort_values('support',ascending= False , inplace= True)

plt.bar(x= list(range(0,11)), height= frequent_itemsets.support[0:11], color=['red','green','grey','yellow','maroon' ,'black'])
plt.xticks(list(range(0,11)),frequent_itemsets.itemsets[0:11], rotation = 10)
plt.xlabel('item-set')
plt.ylabel('support')
plt.show()

#association rules###
rules = association_rules(frequent_itemsets,metric = 'lift', min_threshold=1)

rules.head(20)
rules.sort_values('lift', ascending = False).head(10)

def to_list(i):
    return(sorted(list(i)))

new_rules= rules.antecedents.apply(to_list)+ rules.consequents.apply(to_list)

new_rules = new_rules.apply(sorted)

rules_sets = list(new_rules)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]

index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))
##getting rules w/o redundancy##
rules_no_redundancy = rules.iloc[index_rules, :]

## sort them to list and retrieve top 10 rules##
rules_no_redundancy.sort_values('lift' , ascending = False).head(10)
##################################Problem 3#######################################
import pandas as pd    
from mlxtend.frequent_patterns import apriori , association_rules

##loading dataset#
movies = pd.read_csv('D:/DATA SCIENCE ASSIGNMENT/Datasets_Association Rules/my_movies.csv')
movies = movies.iloc[:,5:] ## wee need only data from column no 5##

movies2 = apriori(movies , min_support=0.002,max_len=4 , use_colnames= True)

##most frequent item based on support##
movies2.sort_values('support' , ascending = False, inplace= True)

import matplotlib.pyplot as plt

plt.bar(x= list(range(0,11)), height= movies2.support[0:11], color=['red','green','grey','yellow','maroon' ,'black'])
plt.xticks(list(range(0,11)),movies2.itemsets[0:11], rotation = 20)
plt.xlabel('item-set')
plt.ylabel('support')
plt.show()

##association rules ##
rules = association_rules(movies2 , metric='lift' , min_threshold=1)
rules.head(20)
rules.sort_values('lift', ascending=False) 

def to_list(i):
    return(sorted(list(i)))

new_rules = rules.antecedents.apply(to_list)+rules.consequents.apply(to_list)

new_rules = new_rules.apply(sorted)

rules_sets = list(new_rules)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]

index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

##getting rules w/o redundancy##
rules_no_redundancy = rules.iloc[index_rules, :]

## sort them to list and retrieve top 10 rules##

rules_no_redundancy.sort_values('lift' , ascending = False).head(10)

#####################################problem 4############################################
import pandas as pd
from mlxtend.frequent_patterns import apriori , association_rules

phonedata = pd.read_csv('D:/DATA SCIENCE ASSIGNMENT/Datasets_Association Rules/myphonedata.csv')
phonedata = phonedata.iloc[:,3 :]

phonedata2 = apriori(phonedata , min_support= 0.02 , use_colnames= True)
##most frequent item based on support##
phonedata2.sort_values('support', ascending= False , inplace = True)

##Plotting##
import matplotlib.pyplot as plt
plt.bar(x= list(range(0,11)), height= phonedata2.support[0:11], color=['red','green','grey','yellow','maroon' ,'black'])
plt.xticks(list(range(0,11)),phonedata2.itemsets[0:11], rotation = 20)
plt.xlabel('item-set')
plt.ylabel('support')
plt.show()

##association rules###
rules = association_rules(phonedata2 , metric= 'lift' , min_threshold= 1)
rules.head(20)
rules.sort_values('lift', ascending=False)

def to_list(i):
    return(sorted(list(i)))

new_rules = rules.antecedents.apply(to_list)+ rules.consequents.apply(to_list)
new_rules = new_rules.apply(sorted)
rules_sets=list(new_rules)
unique_rules_sets= [list(m) for m in set(tuple(i) for i in rules_sets)]

index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

##getting rules w/o redundancy##
rules_no_redundancy = rules.iloc[index_rules , :]
## sort them to list and retrieve top 10 rules##
rules_no_redundancy.sort_values('lift' , ascending = False).head(10)
#############################################Problem 5######################################################
import pandas as pd
from mlxtend.frequent_patterns import apriori , association_rules
retails =[]
with open('D:/DATA SCIENCE ASSIGNMENT/Datasets_Association Rules/transactions_retail1.csv') as f:
    retails = f.read()   
retails = retails.split('\n')    
#lets take 100 datasset##
retails = retails[:100]

retails_list = []
for i in retails:
    retails_list.append(i.split(','))

all_retails_list = [i for item in retails_list for i in item]    
new_all_retails_list = []    
for i in all_retails_list:
    if i!='NA':
        new_all_retails_list.append(i)
from collections import Counter        
item_frequencies = Counter(new_all_retails_list)        
##sorting #
item_frequencies = sorted(item_frequencies.items(),key=lambda x:x[1])
 
##storing frquncies and item in seperate variables##
frequencies = list(reversed([i[1] for i in item_frequencies]))   
items = list(reversed([i[0] for i in item_frequencies]))    

##barplot of top 10##
import matplotlib.pyplot as plt
plt.bar(height=frequencies[0:11], x= list(range(0,11)), color=['red','blue','green''yellow','black'])
plt.xticks(list(range(0,11),),items[0:11])
plt.xlabel('items')
plt.ylabel('count')
plt.show()

##creating dataframe for transaction data##
retails_series = pd.DataFrame(pd.Series(retails_list))

retails_series.columns =['transaction']

x= retails_series['transaction'].str.join(sep= '*').str.get_dummies(sep = '*')
x=x.drop(['NA'], axis = 1)
frequent_itemsets = apriori(x, min_support=0.0075 , max_len=4 , use_colnames= True)

##most frequent itemset based on support ##
frequent_itemsets.sort_values('support' , ascending = False , inplace = True)

import matplotlib.pyplot as plt
plt.bar(x= list(range(0,11)), height= frequent_itemsets.support[0:11], color=['red','green','grey','yellow','maroon' ,'black'])
plt.xticks(list(range(0,11)),frequent_itemsets.itemsets[0:11], rotation = 20)
plt.xlabel('item-set')
plt.ylabel('support')
plt.show()

rules = association_rules(frequent_itemsets , metric = 'lift' , min_threshold= 1)
rules.head(20)
rules.sort_values('lift' , ascending=False).head(10)

def to_list(i):
    return(sorted(list(i))) 

new_rules = rules.antecedents.apply(to_list)+ rules.consequents.apply(to_list)

new_rules = new_rules.apply(sorted)
rules_sets = list(new_rules)
unique_rules_sets= [list(m) for m in set(tuple(i) for i in rules_sets)]

index_rules= []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))
# getting rules without any redudancy     
rules_no_redundancy = rules.iloc[index_rules, :]
# Sorting them with respect to list and getting top 10 rules 
rules_no_redundancy.sort_values('lift' , ascending = False).head(10)




