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
import sklearn
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

mpqa_line = []
subjectivity = []
mpqa_dict = []
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
    mpqa_dict.append({"type" : mpqa_type,
                 "len" : mpqa_len,
                 "word1" : mpqa_word1,
                 "pos1" : mpqa_pos1,
                 "stemmed1" : mpqa_stemmed1,
                 "priorpolarity" : mpqa_priorpolarity
                 })
    
mpqa.close()

# tagging pros & cons
pros_tagged = posTagger.tag_sents(pros)
cons_tagged = posTagger.tag_sents(cons)
###########
w = ""
for sentIdx, sentItem in enumerate(cons['sent']):
     w += (''.join([('' if c in string.punctuation else ' ') + c for c in sentItem]).strip()) + ' '
##############
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

pNFreq = FreqDist(pNounTerms).most_common(20)
cNFreq = FreqDist(cNounTerms).most_common(20)

print(pNFreq)
print(cNFreq)
##########
for word in pros:
    if word not in (sw and ['.', ',', '?', '!', ';', ':', '-', '_', '"', "'"]):
        pNounTerms.append(posTagger.tag(word))

print(pNounTerms)
for line in pros:
    tagged = posTagger.tag(line)
    for word in tagged:
        if str(word[1]).startswith('N') and word[0].lower() not in sw:
            pNounTerms.append(word[0])

for line in cons:
    tagged = posTagger.tag(line)
    for word in tagged:
        if str(word[1]).startswith('N') and word[0].lower() not in sw:
            cNounTerms.append(word[0])
####################

# Tokenizing the document
textSentence = nltk.sent_tokenize(camera_reviews)
textWord = nltk.word_tokenize(camera_reviews)

# Parse and POS-tag the sentences
parsedSentence = parser.raw_parse_sents(textSentence)

for review in camera_reviews:
    print(review.sents)

#
parsedStr = []
for line in parsedSentence:
    s = ''
    for sentence in line:
        # for a in sentence:
        # if sentence.label() == 'NP':
        #     print(sentence.pos())
        s += str(sentence)
        # s.append(sentence)
        # sentence.draw()

    parsedStr.append(s)

nounPhraseFn = []
nounPhrase = []


def traverse_tree(tree):
    if tree.label() == 'NP':
        for subtree in tree:
            if str(subtree.label()).startswith('N'):
                nounPhraseFn.extend(subtree.leaves())
    for subtree in tree:
        if type(subtree) == nltk.tree.Tree:
            traverse_tree(subtree)

    return nounPhraseFn


def pre_process(sentence):
    sentence = sentence.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    filtered_words = filter(lambda token: token not in stopwords.words('english'), tokens)
    return " ".join(filtered_words)


for i in parsedStr:
    nounPhrase.extend(traverse_tree(Tree.fromstring(i)))
    nounPhraseFn = []

wordDist = FreqDist(nounPhrase)
# print(wordDist.most_common(10))

print("--- %s seconds ---" % (time.time() - start_time))
