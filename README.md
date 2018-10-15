# d18

#bayes-test
import numpy as np

def loadDataSet():                                                                                #数据库
    List=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    Labels = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return List,Labels
def createVocabList(dataset):                                                                  #构建词汇表
    vocabSet = set([])
    for line in dataset:
        vocabSet = vocabSet | set(line)
    return list(vocabSet)
X,y = loadDataSet()
VocabList = createVocabList(X)
print(VocabList)
def words2vec(vocabList,inputSet):                                                             #文档词汇计入向量
    vec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            vec[vocabList.index(word)] = 1
    return vec
vec1 = words2vec(VocabList,X[0])
vec2 = words2vec(VocabList,X[1])
vec3 = words2vec(VocabList,X[2])
vec4 = words2vec(VocabList,X[3])
vec5 = words2vec(VocabList,X[4])
vec6 = words2vec(VocabList,X[5])
vec = np.array([vec1,
                vec2,
                vec3,
                vec4,
                vec5,
                vec6])
#print(vec)
cla = BernoulliNB()
cla.fit(vec,y)
test = [['maybe','dalmation', 'is', 'stupid', 'garbage','stop']]
test1 = words2vec(VocabList,test)
print(cla.predict([test1]).reshape(1,-1))

#bayes-email
import re
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn import metrics

# f = open('email/ham/1.txt').read()
# print(f)

def textParse(string):
    text = re.split(r'\W*',string)
    return [word.lower() for word in text if len(word)>2]
def createVocabList(dataset):
    VocabSet = set([])
    for line in dataset:
        VocabSet = VocabSet | set(line)
    return list(VocabSet)
def words2vec(vocabList,inputSet):
    vec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            vec[vocabList.index(word)] = 1
    return vec
def spamTest():
    docList = []
    labelList = []
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' %i).read())
        docList.append(list(wordList))
        labelList.append(1)
        wordList = textParse(open('email/ham/%d.txt' %i).read())
        docList.append(list(wordList))
        labelList.append(0)
    vocabList = createVocabList(docList)
    return docList,labelList,vocabList
X,y,vocabList= spamTest()
vec = []
#print(len(X))
for i in range(50):
    vec.append(words2vec(vocabList,X[i]))
#print(words2vec(vocabList,X))
#print(X)
#print(y)
X_train,X_test,y_train,y_test = train_test_split(vec,y,test_size=.3,random_state=1)
cla = BernoulliNB()
cla.fit(X_train,y_train)
y_pred = cla.predict(X_test)
print(cla.predict(X_test))
print(metrics.accuracy_score(y_test,y_pred))
