import nltk
from htmldom import htmldom
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.wsd import lesk


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return ''


lemmatizer = WordNetLemmatizer()
sw = set(stopwords.words("english"))

query = input("Enter your desired query: ")
queryWords = word_tokenize(query)
querySentences = sent_tokenize(query)

queryTagged = []
for sent in querySentences:
    queryTagged.append(nltk.pos_tag(nltk.word_tokenize(sent)))

cleanQuery = []
querySentCount = 0
for sent in queryTagged:
    for tagSet in sent:
        if tagSet[0] not in sw and get_wordnet_pos(tagSet[1]) not in '':
            cleanQuery.append(
                {'word': lemmatizer.lemmatize(tagSet[0], pos=get_wordnet_pos(tagSet[1])).lower(),
                 'sentNum': querySentCount, 'pos': get_wordnet_pos(tagSet[1])})
    querySentCount = querySentCount + 1

querySynsets = []
for word in cleanQuery:
    querySynsets.append(wn.synsets(word['word']))

queryWordSense = []
for wordSet in cleanQuery:
    wsd = lesk(querySentences[wordSet['sentNum']], wordSet['word'], wordSet['pos'])
    if wsd is not None:
        queryWordSense.append(wsd)

queryHypos = []
queryHypers = []
for syns in queryWordSense:
    if syns is not None:
        queryHypos.append(syns.hyponyms())
        queryHypers.append(syns.hypernyms())

querySynsClean = []
for seti in querySynsets:
    for word in seti:
        querySynsClean.append(word.lemmas()[0].name().replace('_', ' '))

queryHyposClean = []
for seti in queryHypos:
    for word in seti:
        queryHyposClean.append(word.lemmas()[0].name().replace('_', ' '))

queryHypersClean = []
for seti in queryHypers:
    for word in seti:
        queryHypersClean.append(word.lemmas()[0].name().replace('_', ' '))

dom = htmldom.HtmlDom("https://en.wikipedia.org/wiki/Ford_Motor_Company").createDom()
content = dom.find("div[id=mw-content-text]").text()
contentWords = word_tokenize(content)
contentSentences = sent_tokenize(content)

contentTagged = []
for sent in contentSentences:
    contentTagged.append(nltk.pos_tag(nltk.word_tokenize(sent)))

cleanContent = []
contentSentCount = 0
for sent in contentTagged:
    for tagSet in sent:
        if tagSet[0] not in sw and get_wordnet_pos(tagSet[1]) not in '':
            cleanContent.append({'word': lemmatizer.lemmatize(tagSet[0], pos=get_wordnet_pos(tagSet[1])).lower(),
                                 'sentNum': contentSentCount, 'pos': get_wordnet_pos(tagSet[1])})
    contentSentCount = contentSentCount + 1

wordsCount = 0
for w1 in cleanQuery:
    for w2 in cleanContent:
        if w1['word'] == w2['word']:
            wordsCount = wordsCount + 1

synsCount = 0
for w1 in querySynsClean:
    for w2 in cleanContent:
        if w1 == w2['word']:
            synsCount = synsCount + 1

hyposCount = 0
for w1 in queryHyposClean:
    for w2 in cleanContent:
        if w1 == w2['word']:
            hyposCount = hyposCount + 1

hypersCount = 0
for w1 in queryHypersClean:
    for w2 in cleanContent:
        if w1 == w2['word']:
            hypersCount = hypersCount + 1

alpha = 0.47
beta = 0.84
gamma = 0.7

ttf = wordsCount + (alpha * hypersCount) + (beta * hyposCount) + (gamma * synsCount)

print('query-page similarity with expansion: ', ttf)
print('query-page similarity without expansion: ', wordsCount)
