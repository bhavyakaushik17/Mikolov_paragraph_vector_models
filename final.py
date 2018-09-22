# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 15:17:17 2018

@author: Bhavya
"""

# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from gensim.models import Word2Vec

# numpy
import numpy
import pandas as pd

#plot
import matplotlib.pyplot as plt

# shuffle
from random import shuffle

# logging
import logging
import os.path
import sys

# classifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import binarize

# Setting up logger
program = os.path.basename(sys.argv[0])
log = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
log.info("running %s" % ' '.join(sys.argv))
######################################################################################################################################################
# labelled sentences class to assist and simplify the doc2vec training corpora preprocessing, logging
class LabeledLineSentence(object):

    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(
                        utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences
################################################################################################################################################################
### Loading the data from already preprocessed files
#shorter version 1% of full version
sources = {'test-neg.txt':'TEST_NEG', 'test-pos.txt':'TEST_POS', 'train-neg.txt':'TRAIN_NEG', 'train-pos.txt':'TRAIN_POS'}
#sources = {'test-neg.txt':'TEST_NEG', 'test-pos.txt':'TEST_POS', 'train-neg.txt':'TRAIN_NEG', 'train-pos.txt':'TRAIN_POS'}
### Note the order of reading the input, test neg, then test pos, then train neg, then train pos
sentences = LabeledLineSentence(sources)
corp=sentences.to_array()

#number of words in the dataset
size_dataset=0
for i in range(len(corp)):
    size_dataset = size_dataset + len(corp[i][0])
print(size_dataset)
#############################################################################################################################
naam1 = ["model_size100","model_size300","model_size700"]
naam2 = ["model_sub-7","model_sub-6","model_sub-5","model_sub-4","model_sub-3","model_sub-2","model_sub-1","model_sub0"]
naam3 = ["model_neg0","model_neg2","model_neg5","model_neg20","model_neg100"]
naam4 = ["model_epoch0","model_epoch1","model_epoch5","model_epoch20"]

s=naam2+naam3+naam4
score=numpy.zeros(len(s)+len(naam1))
c=numpy.zeros((2,2))
conf=[]
for i in range(len(s)+len(naam1)):
    conf.append(c)

dim =[100, 300, 700]
for x in range(len(naam1)):

    model = Doc2Vec.load('./'+str(naam1[x])+'.d2v')
    log.info('Sentiment')

    train_arrays_d2v = numpy.zeros((40000, dim[x]))
    train_labels_d2v = numpy.zeros(40000)
    for i in range(12500):
        prefix_train_pos = 'TRAIN_POS_' + str(i)
        prefix_train_neg = 'TRAIN_NEG_' + str(i)
        train_arrays_d2v[i] = model.docvecs[prefix_train_pos]
        train_arrays_d2v[12500 + i] = model.docvecs[prefix_train_neg]
        train_labels_d2v[i] = 1
        train_labels_d2v[12500 + i] = 0
    for i in range(7500):
        prefix_test_pos = 'TEST_POS_' + str(i)
        prefix_test_neg = 'TEST_NEG_' + str(i)
        train_arrays_d2v[25000 + i] = model.docvecs[prefix_test_pos]
        train_arrays_d2v[32500 + i] = model.docvecs[prefix_test_neg]
        train_labels_d2v[25000 + i] = 1
        train_labels_d2v[32500 + i] = 0
    log.info(train_labels_d2v)

    test_arrays_d2v = numpy.zeros((10000, dim[x]))
    test_labels_d2v = numpy.zeros(10000)
    for i in range(5000):
        prefix_test_pos = 'TEST_POS_' + str(7500 + i)
        prefix_test_neg = 'TEST_NEG_' + str(7500 + i)
        test_arrays_d2v[i] = model.docvecs[prefix_test_pos]
        test_arrays_d2v[5000 + i] = model.docvecs[prefix_test_neg]
        test_labels_d2v[i] = 1
        test_labels_d2v[5000 + i] = 0
    log.info(test_labels_d2v)

    #logistic regression classifier
    log.info('Fitting')
    classifier = LogisticRegression()
    classifier.fit(train_arrays_d2v, train_labels_d2v)
    #Result
    log.info(classifier.score(test_arrays_d2v, test_labels_d2v))
    score[x]=classifier.score(test_arrays_d2v, test_labels_d2v)
    ################################################################################################################################################################
    #CONFUSION MATRIX
    test_pred_labels_d2v = classifier.predict(test_arrays_d2v)
    test_pred_prob_d2v = classifier.predict_proba(test_arrays_d2v)[:, 1]
    ##CONFUSION MATRIX
    print("\nConfusion matrix:\n")
    confusion = metrics.confusion_matrix(test_labels_d2v, test_pred_labels_d2v)
    print(confusion)
    conf[x]=confusion


for x in range(len(naam1) , len(naam1)+len(s)):

    model = Doc2Vec.load('./'+str(s[x-3])+'.d2v')
    log.info('Sentiment')

    train_arrays_d2v = numpy.zeros((40000, 100))
    train_labels_d2v = numpy.zeros(40000)
    for i in range(12500):
        prefix_train_pos = 'TRAIN_POS_' + str(i)
        prefix_train_neg = 'TRAIN_NEG_' + str(i)
        train_arrays_d2v[i] = model.docvecs[prefix_train_pos]
        train_arrays_d2v[12500 + i] = model.docvecs[prefix_train_neg]
        train_labels_d2v[i] = 1
        train_labels_d2v[12500 + i] = 0
    for i in range(7500):
        prefix_test_pos = 'TEST_POS_' + str(i)
        prefix_test_neg = 'TEST_NEG_' + str(i)
        train_arrays_d2v[25000 + i] = model.docvecs[prefix_test_pos]
        train_arrays_d2v[32500 + i] = model.docvecs[prefix_test_neg]
        train_labels_d2v[25000 + i] = 1
        train_labels_d2v[32500 + i] = 0
    log.info(train_labels_d2v)

    test_arrays_d2v = numpy.zeros((10000, 100))
    test_labels_d2v = numpy.zeros(10000)
    for i in range(5000):
        prefix_test_pos = 'TEST_POS_' + str(7500 + i)
        prefix_test_neg = 'TEST_NEG_' + str(7500 + i)
        test_arrays_d2v[i] = model.docvecs[prefix_test_pos]
        test_arrays_d2v[5000 + i] = model.docvecs[prefix_test_neg]
        test_labels_d2v[i] = 1
        test_labels_d2v[5000 + i] = 0
    log.info(test_labels_d2v)

    #logistic regression classifier
    log.info('Fitting')
    classifier = LogisticRegression()
    classifier.fit(train_arrays_d2v, train_labels_d2v)
    #Result
    log.info(classifier.score(test_arrays_d2v, test_labels_d2v))
    score[x]=classifier.score(test_arrays_d2v, test_labels_d2v)
    ################################################################################################################################################################
    #CONFUSION MATRIX
    test_pred_labels_d2v = classifier.predict(test_arrays_d2v)
    test_pred_prob_d2v = classifier.predict_proba(test_arrays_d2v)[:, 1]
    ##CONFUSION MATRIX
    print("\nConfusion matrix:\n")
    confusion = metrics.confusion_matrix(test_labels_d2v, test_pred_labels_d2v)
    print(confusion)
    conf[x]=confusion

t=list(zip(score,conf))
naam3 = ["model_neg000","model_neg002","model_neg005","model_neg020","model_neg100"]
naam4 = ["model_epoch00","model_epoch01","model_epoch05","model_epoch20"]
s =naam1+naam2+naam3+naam4
z=list(zip(s,t))

d = {key: value for (key, value) in z}
df = pd.DataFrame(data=d,index=['accuracy score','confusion matrix'])

writer = pd.ExcelWriter('corrected.xlsx')
df.to_excel(writer,sheet_name='corrected',  merge_cells=True, startrow=0, startcol=0)
writer.save()

####################################################################################################################################

y=df.values[0]

epochs=['0','1','5','20']
z=numpy.arange(4)

plt.plot(z, y[0:4],'-d')
plt.title('accuracy vs epochs')
plt.xlabel('number of epochs')
plt.ylabel('accuracy score')
plt.ylim([0.0, 1.0])
plt.xticks(z, epochs)
plt.grid(True)
for i in range(4):
    plt.annotate("("+epochs[i]+","+str(y[i])+")",xy=(z[i],y[i]))


neg=['0','2','5','20','100']
z=numpy.arange(5)

plt.plot(z, y[4:9],'-d')
plt.title('accuracy vs "n"')
plt.xlabel('negative sampling parameter "n"')
plt.ylabel('accuracy score')
plt.ylim([0.0, 1.0])
plt.xticks(z,neg)
plt.grid(True)
for i in range(4,9):
    plt.annotate("("+neg[i-4]+","+str(y[i])+")",xy=(z[i-4],y[i]))


size=['100','300','700']
z=numpy.arange(3)

plt.plot(z, y[9:12],'-d')
plt.title('accuracy vs dimension of vector')
plt.xlabel('dimension of vector')
plt.ylabel('accuracy score')
plt.ylim([0.0, 1.0])
plt.xticks(z,size)
plt.grid(True)
for i in range(9,12):
    plt.annotate("("+size[i-9]+","+str(y[i])+")",xy=(z[i-9],y[i]))

sub = ['1e-7', '1e-6', '1e-5', '1e-4', '1e-3', '1e-2', '1e-1', '1']
z=numpy.arange(8)

plt.plot(z, y[12:20],'-d')
plt.title('accuracy vs "t"')
plt.xlabel('subsampling parameter "t"')
plt.ylabel('accuracy score')
plt.ylim([0.0, 1.0])
plt.xticks(z,sub)
plt.grid(True)
for i in range(12,20):
    plt.annotate("("+sub[i-12]+","+str(y[i])+")",xy=(z[i-12],y[i]))