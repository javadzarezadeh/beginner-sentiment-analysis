from nltk.corpus import product_reviews_1
from nltk.parse.stanford import StanfordParser
import string
import nltk
from nltk.util import breadth_first
from nltk.tree import Tree

# Define Stanford Parser
parser = StanfordParser(
    path_to_jar='./stanford-parser-full-2015-12-09/stanford-parser.jar',
    path_to_models_jar='./stanford-english-corenlp-2016-01-10-models.jar',
    model_path='./stanford-english-corenlp-2016-01-10-models/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz'
)

camera_reviews = product_reviews_1.reviews('Canon_G3.txt')

#print(nltk.tag.pos_tag_sents(camera_reviews[0].sents()))

# Tokenizing the document
#textSentence = nltk.sent_tokenize(camera_reviews)
#textWord = nltk.word_tokenize(camera_reviews)

# Parse and POS-tag the sentences
#parsedSentence = parser.raw_parse_sents(textSentence)

a = camera_reviews[0].sents()[0]
print(a)

nounPhraseFn = []
nounPhrase = []

b = ''
for word in a:
    b += word + ' '
print(b)
parsedSentence = parser.raw_parse_sents(b)

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
    
#traverse_tree(parsedStr)


def traverse_tree(tree):
    if tree.label() == 'NP':
        for subtree in tree:
            if str(subtree.label()).startswith('N'):
                nounPhraseFn.extend(subtree.leaves())
    for subtree in tree:
        if type(subtree) == nltk.tree.Tree:
            traverse_tree(subtree)

    return nounPhraseFn
    
#traverse_tree(a)