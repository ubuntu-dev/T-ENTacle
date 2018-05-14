
# to help create semantic-signatures for the contents of a cell. Then match them with adjacent cells to detect tabular content from a csv.
# magnifying glass, tree matching, glo-op
# for each token get a signature

import pandas as pd
import jellyfish
import numpy as np
from collections import Counter

def find_ngrams(input_list, n):
  return zip(*[input_list[i:] for i in range(n)])


def createsign(string):
    def decide(arr):
        if arr[0]=='A' and len(arr)>2:
            return 'W'
        if arr[0]=='n' and len(arr)>2:
            return 'N'
        else:
            return ''.join(arr)

    signature=''
    string=str(string)
    for ch in string:
        if ch.isalpha():
            signature+='A'
        elif ch.isnumeric():
            signature+='n'
        else:
            signature+=ch

    # condense the patterns to mark words
    condensed_signature=''
    buffer=[]
    for i, ch in enumerate(signature):
        buffer.append(ch)
        if i!=len(signature)-1 and ch!=signature[i+1]:
            condensed_signature+=decide(buffer)
            buffer=[]
        elif i==len(signature)-1:
            condensed_signature+=decide(buffer)


    return condensed_signature

#get cosine similarity between two document vectors
def get_cosine(vec1, vec2):
    import math
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def text_to_vector(text):
    return Counter(text)

#get cosine similarity between texts
def CosineSim(text1,text2):
    vec1=text_to_vector(text1)
    vec2=text_to_vector(text2)
    dist=get_cosine(vec1,vec2)
    return dist



def levenshtein_similarity(s, t):
    """ Levenshtein Similarity """
    Ns = len(s);
    Nt = len(t);
    lev_sim = 1.0 - (jellyfish.levenshtein_distance(s, t)) / float(max(Ns, Nt))
    return lev_sim


def jaro_winkler_similarity(s, t):
    """ Jaro-Winkler Similarity """
    jw_sim = jellyfish.jaro_winkler(s, t)
    return jw_sim

def similarity(a,b):
    a_sign=createsign(a)
    b_sign=createsign(b)
    return levenshtein_similarity(a_sign,b_sign)

def similarity2(a,b):
    a_sign=list(createsign(a))
    b_sign=list(createsign(b))
    return CosineSim(a_sign,b_sign)


def simpleNN(column,clusternum,threshold=0.9):
    clusters=[[column[0]]] #initialize
    clusternums=[clusternum]

    for i in range(1,len(column)):
        if similarity2(column[i],column[i-1])>=threshold:
            clusters[-1].append(column[i])
        else:
            clusternum+=1
            clusters.append([column[i]])

        clusternums.append(clusternum)

    return clusternums,clusters,clusternum


# examples:

string='55 P -  PUMPING  (NON-FRAC)'
print(''.join(createsign(string)))

# read table:

csv=pd.read_csv('../../docs/demo2.png.csv')
clusternum=-1
table_matrix=[]
for i,column in enumerate(csv.keys()):
    clusternums, clusters, clusternum=simpleNN(csv[column],clusternum+1)
    csv[column]=csv[column].apply(lambda x:createsign(x))
    table_matrix.append(clusternums)


table_matrix=np.array(np.transpose(table_matrix))
print(table_matrix)
print(csv)
