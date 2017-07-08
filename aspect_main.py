from nltk.corpus import pros_cons, product_reviews_1, stopwords
from nltk.parse.stanford import StanfordParser
import nltk
from nltk.tree import Tree
from nltk import FreqDist
from nltk.text import Text
from nltk.tag.stanford import StanfordPOSTagger
from nltk.tokenize import RegexpTokenizer, sent_tokenize
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
import string
import time
import sys

start_time = time.time()

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

pNounTerms = []
cNounTerms = []
# extract nouns from pros
for sentence in pros_tagged:
    for word in sentence:
        if word[0] not in (sw and string.punctuation) and str(word[1]).startswith('N'):
            pNounTerms.append(word[0].lower())

# extract nouns from cons
for sentence in cons_tagged:
    for word in sentence:
        if word[0] not in (sw and string.punctuation) and str(word[1]).startswith('N'):
            cNounTerms.append(word[0].lower())

pcNounTerms = pNounTerms + cNounTerms
pcNFreq = FreqDist(pcNounTerms).most_common(1000)
word_features = [w[0] for w in pcNFreq]
'''
# Tokenizing the document
textSentence = nltk.sent_tokenize(camera_reviews)
textWord = nltk.word_tokenize(camera_reviews)

# Parse and POS-tag the sentences
parsedSentence = parser.raw_parse_sents(textSentence)
'''


# make tuples of strings from list of sentence words
def listToTuplesOfStr(listSent):
    sentenceList = []
    for sentence in listSent:
        strSent = ''
        for word in sentence:
            strSent += (str(word) + ' ')
        sentenceList.append(strSent)
    return tuple(sentenceList)


# traverse trees
# nounPhraseFn = []
nounPhrase = []


def traverse_tree(tree):
    if tree.label() == 'NP':
        for subtree in tree:
            if str(subtree.label()).startswith('N'):
                nounPhrase.extend(subtree.leaves())
    for subtree in tree:
        if type(subtree) == nltk.tree.Tree:
            traverse_tree(subtree)

    return nounPhrase


# making array of parsed sentences
parsedReviews = []
for r in camera_reviews:
    sentText = listToTuplesOfStr(r.sents())
    parsedReviews.extend(parser.raw_parse_sents(sentText))

# extracting nouns
textNouns = []
for rev in parsedReviews:
    for line in rev:
        for sentence in line:
            textNouns.extend(traverse_tree(sentence))
# sentence.draw()
noun_num = FreqDist(textNouns)

# featureSet = {}
# for w in word_features:
#    featureSet[w] = noun_num[w]

wf = []
for w in noun_num:
    temp = []
    for x in word_features:
        if x == w:
            temp.append(noun_num[x])
        else:
            temp.append(0)
    wf.append(temp)

train = np.array(wf)
clf = OneClassSVM()
clf.fit(train)
pred_train = clf.predict(train)
print("--- %s seconds ---" % (time.time() - start_time))
print('hi')
