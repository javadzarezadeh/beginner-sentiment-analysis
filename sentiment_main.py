from nltk.corpus import pros_cons
from nltk.corpus import product_reviews_1
from nltk.parse.stanford import StanfordParser
import nltk
from nltk.tree import Tree
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.tag.stanford import StanfordPOSTagger
from nltk.tokenize import RegexpTokenizer
import pandas as pd
import numpy as np
import sklearn
import string
import time
import sys

start_time = time.time()
StanfordParser()
# Define Stanford Parser
parser = StanfordParser(
    path_to_jar='./stanford-parser-full-2015-12-09/stanford-parser.jar',
    path_to_models_jar='./stanford-english-corenlp-2016-01-10-models.jar',
    model_path='./stanford-english-corenlp-2016-01-10-models/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz'
)

# Define Stanford POS-tagger
posTagger = StanfordPOSTagger(
    path_to_jar='./stanford-postagger-2015-12-09/stanford-postagger-3.6.0.jar',
    model_filename='./stanford-postagger-2015-12-09/models/english-bidirectional-distsim.tagger')

# Define stopwords
sw = set(stopwords.words("english"))

# Define document
camera_reviews = product_reviews_1.reviews('Canon_G3.txt')

# Defining Pros and Cons
pros = pros_cons.sents(categories='Pros')
cons = pros_cons.sents(categories='Cons')

# tagging pros & cons
pros_tagged = posTagger.tag_sents(pros)
cons_tagged = posTagger.tag_sents(cons)

mpqa_line = []
subjectivity = []
mpqa_dict = []
mpqa_words = []
# Define and work with MPQA sentiment lexicon
mpqa = open("./subjectivity_clues_hltemnlp05/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff", "r")
for line in mpqa.readlines():
    mpqa_line.append(line.split())

for item in mpqa_line:
    for subItem in item:
        if subItem.split("=")[0] == "type":
            mpqa_type = subItem.split("=")[1]
        if subItem.split("=")[0] == "len":
            mpqa_len = subItem.split("=")[1]
        if subItem.split("=")[0] == "word1":
            mpqa_word1 = subItem.split("=")[1]
        if subItem.split("=")[0] == "pos1":
            mpqa_pos1 = subItem.split("=")[1]
        if subItem.split("=")[0] == "stemmed1":
            mpqa_stemmed1 = subItem.split("=")[1]
        if subItem.split("=")[0] == "priorpolarity":
            mpqa_priorpolarity = subItem.split("=")[1]
    mpqa_dict.append({"type": mpqa_type,
                      "len": mpqa_len,
                      "word1": mpqa_word1,
                      "pos1": mpqa_pos1,
                      "stemmed1": mpqa_stemmed1,
                      "priorpolarity": mpqa_priorpolarity
                      })
    mpqa_words.append(mpqa_word1)

mpqa.close()

# for p in pros:
#    for item in p:
##        if item in mpqa_words:
#        for i in mpqa_dict:
#            if item == i['word1']:
#                print(i['word1'] + ' ' + i['priorpolarity'])


pAdj = []
cAdj = []
# extract nouns from pros
for sentence in pros_tagged:
    for word in sentence:
        if word[0] not in (sw and string.punctuation) and str(word[1]).startswith('J'):
            pAdj.append(word[0].lower())

# extract nouns from cons
for sentence in cons_tagged:
    for word in sentence:
        if word[0] not in (sw and string.punctuation) and str(word[1]).startswith('J'):
            cAdj.append(word[0].lower())

pcAdj = pAdj + cAdj
pcAdjFreq = FreqDist(pcAdj).most_common(1000)
word_features = [w[0] for w in pcAdjFreq]

# wf = []
# for w in noun_num:
#    temp = []
#    for x in word_features:
#        if x == w:
#            temp.append(noun_num[x])
#        else:
#            temp.append(0)
#    wf.append(temp)


features = []
for r in camera_reviews:
    r_feature = np.zeros(1000)
    for sentence in r.sents():
        for word in sentence:
            #            mpqa_index = [i for i,_ in enumerate(mpqa_dict) if _['word1'] == word][0]
            #            mpqa_index = mpqa_dict.index(filter(lambda n: n.get('word1') == word, mpqa_dict)[0])
            mpqa_index = next((index for (index, d) in enumerate(mpqa_dict) if d["word1"] == word), None)
            if mpqa_index is not None:
                if word in word_features:
                    word_index = word_features.index(word)
                    if (mpqa_dict[mpqa_index])['priorpolarity'] == 'neutral':
                        r_feature[word_index] += 0.0
                    elif (mpqa_dict[mpqa_index])['priorpolarity'] == 'negative':
                        r_feature[word_index] -= 1.0
                    elif (mpqa_dict[mpqa_index])['priorpolarity'] == 'positive':
                        r_feature[word_index] += 1.0
                        #                       r_feature[word_index] += (mpqa_dict[mpqa_index])['priorpolarity']
                        #                else:
                        #                    r_feature[word_index] += 0
    features.append(r_feature)

# for p in pros:
#    for item in p:
#        if item in mpqa_words:
#            print(item)
# for c in cons:

# for item in mpqa_dict:
#    if 'absurdly' == item['word1']:
#        print('a')

print("--- %s seconds ---" % (time.time() - start_time))
