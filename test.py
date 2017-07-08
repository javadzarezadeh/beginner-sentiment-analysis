from nltk.corpus import product_reviews_1
from nltk.parse.stanford import StanfordParser
import string
import nltk
from nltk.util import breadth_first
from nltk.tree import Tree

parser = StanfordParser(
    path_to_jar='./stanford-parser-full-2015-12-09/stanford-parser.jar',
    path_to_models_jar='./stanford-english-corenlp-2016-01-10-models.jar',
    model_path='./stanford-english-corenlp-2016-01-10-models/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz'
)

camera_reviews = product_reviews_1.reviews('Canon_G3.txt')

review_sents = []
parsedSentence = []
for reviewId, reviewItm in enumerate(camera_reviews):
    w = ""
    for sent in reviewItm.sents():
        # for word in sent:
        w += (''.join([('' if c in string.punctuation else ' ') + c for c in sent]).strip()) + ' '
        review_sents.append(w)
# for sent in review_sents:
    # print(parsedSentence.append(parser.raw_parse_sents(sent)))
print(breadth_first(parser.raw_parse_sents(review_sents[1])))
# parsedStr = []
# for r in review_sents:
#     for s in parser.raw_parse_sents(nltk.sent_tokenize(r)):
#         s = ''
#         for w in s:
#             s += str(w)
#         parsedStr.append(s)
#print(nltk.util.breadth_first(review_sents[1]))
# print(parsedStr)
# print(parser.raw_parse_sents(review_sents[1]))


# parsedStr = []
# for line in parsedSentence:
#     s = ''
#     for sentence in line:
#         s += str(sentence)
#         # s.append(sentence)
#         # sentence.draw()
#
#     parsedStr.append(s)
# nounPhraseFn = []
# nounPhrase = []
#
#
# def traverse_tree(tree):
#     if tree.label() == 'NP':
#         for subtree in tree:
#             if str(subtree.label()).startswith('N'):
#                 nounPhraseFn.extend(subtree.leaves())
#     for subtree in tree:
#         if type(subtree) == nltk.tree.Tree:
#             traverse_tree(subtree)
#
#     return nounPhraseFn
#
#
# for i in parsedStr:
#     nounPhrase.extend(traverse_tree(Tree.fromstring(i)))
#     nounPhraseFn = []
#
# print(nounPhrase)
