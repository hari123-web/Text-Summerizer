# nltk (Natural language toolkit)
# uses:
#1> programs for symbolic and statistical natural language processing(NLP)
#2> Lexical analysis : word and text tokenizer
#3> n-gram and collocations
#4> part-of-speech tagger
#5> tree model and text chunker for capturing
#6> Named-entity recognition
# Mearure similarity between two sentences using cosine similarity
# measure of similarity between two non zero vectors of an inner product space that - 
# measures the cosine of the angle between them.
#  similarity = (A.B)/(||A||.||B||)  where A and B are vectors


import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance   #use to measur similarities between 2 sentences
import numpy as np
import networkx as nx

def read_article(file_name):
    file = open(file_name,"r")
    filedata = file.readlines()
    article = filedata[0].split(". ")
    sentences=[]
    for sentence in article:
        sentences.append(sentence.replace("[^a-zA-Z]"," ").split(" "))
    sentences.pop()
    return sentences


def sentence_similarity(sent1,sent2,stopwords=None):
    if stopwords is None:
        stopwords=[]
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    all_words = list(set(sent1+sent2))  #set to identify unique words

    vector1= [0]*len(all_words)    
    vector2= [0]*len(all_words)
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)]+=1
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)]+=1
    return 1-cosine_distance(vector1,vector2)

def gen_sim_matrix(sentences,stop_words):
    similarity_matrix = np.zeros((len(sentences),len(sentences)))
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1==idx2:
                continue
            similarity_matrix[idx1][idx2]=sentence_similarity(sentences[idx1],sentences[idx2],stop_words)

    return similarity_matrix        

def generate_summary(file_name,top_n=12):
    stop_words=stopwords.words('english')
    summarize_text=[]
    sentences = read_article(file_name)
    sentence_similarity_matrix = gen_sim_matrix(sentences,stop_words)
    sentences_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentences_similarity_graph)
    ranked_sentence =  sorted(((scores[i],s) for i,s in enumerate(sentences)),reverse=True)
    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentence[i][1]))
    print("Summary \n",". ".join(summarize_text))

generate_summary("demo_text1.txt",4)

            

