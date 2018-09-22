# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 23:05:02 2018

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

# plot
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

#statistics
from scipy import stats

# datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit

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


######################################################################################################################################
model = Doc2Vec.load('./model_epoch5.d2v')
log.info('Sentiment')

x = numpy.zeros((50000,100))
y = numpy.zeros(50000)

for i in range(12500):
    x[i] = model.docvecs['TEST_NEG_'+str(i)]
    x[12500+i] = model.docvecs['TRAIN_NEG_'+str(i)]
    x[25000+i] = model.docvecs['TEST_POS_'+str(i)]
    x[37500+i] = model.docvecs['TRAIN_POS_'+str(i)]
    y[i] = 0
    y[12500+i] = 0
    y[37500+i] = 1
    y[25000+i] = 1

skf = StratifiedKFold(n_splits=10)
classifier = LogisticRegression()

acc=numpy.zeros(10)
auc=numpy.zeros(10)
i = 0
for train_index, test_index in skf.split(x,y):
    print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    classifier.fit(x_train, y_train)
    acc[i] = classifier.score(x_test, y_test)
    p = classifier.predict_proba(x_test)[:, 1]
    auc[i] = metrics.roc_auc_score(y_test, p)
    i = i + 1

t=list(zip(acc,auc))
z=list(zip(numpy.arange(10),t))
d= {key: value for (key, value) in z}
df = pd.DataFrame(data=d,index=['accuracy score','auc score'])
writer = pd.ExcelWriter('comparison.xlsx')
df.to_excel(writer,sheet_name='pvdbow+sk')
###############################################################################################################################
"""
model2 = Doc2Vec(dm =1, dbow_words=0, min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=8)
model2.build_vocab(corp)
for epoch in range(5):
    log.info('Epoch %d' % epoch)
    model2.train(sentences.sentences_perm(),
                total_examples=model2.corpus_count,
                epochs=model2.iter,)
model2.save('./imdb_pvdm.d2v')
"""

model2 = Doc2Vec.load('./imdb_pvdm.d2v')
log.info('Sentiment')

x = numpy.zeros((50000,100))
y = numpy.zeros(50000)

for i in range(12500):
    x[i] = model2.docvecs['TEST_NEG_'+str(i)]
    x[12500+i] = model2.docvecs['TRAIN_NEG_'+str(i)]
    x[25000+i] = model2.docvecs['TEST_POS_'+str(i)]
    x[37500+i] = model2.docvecs['TRAIN_POS_'+str(i)]
    y[i] = 0
    y[12500+i] = 0
    y[37500+i] = 1
    y[25000+i] = 1

i = 0
for train_index, test_index in skf.split(x,y):
    print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    classifier.fit(x_train, y_train)
    acc[i] = classifier.score(x_test, y_test)
    p = classifier.predict_proba(x_test)[:, 1]
    auc[i] = metrics.roc_auc_score(y_test, p)
    i = i + 1

t=list(zip(acc,auc))
z=list(zip(numpy.arange(10),t))
d= {key: value for (key, value) in z}
df = pd.DataFrame(data=d,index=['accuracy score','auc score'])
df.to_excel(writer,sheet_name='pvdm')

############################################################################################################################
w2v_corp= []
for i in range(len(corp)):
    w2v_corp.append(corp[i][0])
'''
#training model
model1 = Word2Vec(w2v_corp, min_count=1, size = 100)
#Saving the model
model1.save('./imdb.w2v')
#doc vectors: v[i] ; averaging out word vectors for each document
'''
model1 = Word2Vec.load("./imdb.w2v")
v=numpy.zeros((len(corp), 100))
for i in range(len(w2v_corp)):
    for word in w2v_corp[i]:
        v[i]=v[i]+model1[word]/len(w2v_corp[i])
#end of training
x = numpy.zeros((50000,100))
y = numpy.zeros(50000)

for i in range(12500):
    x[i] = v[i]
    x[12500+i] = v[12500+i]
    x[25000+i] = v[25000+i]
    x[37500+i] = v[37500+i]
    y[i] = 0
    y[12500 + i] = 1
    y[25000+i] = 0
    y[37500 + i] = 1

i = 0
for train_index, test_index in skf.split(x,y):
    print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    classifier.fit(x_train, y_train)
    acc[i] = classifier.score(x_test, y_test)
    p = classifier.predict_proba(x_test)[:, 1]
    auc[i] = metrics.roc_auc_score(y_test, p)
    i = i + 1

t=list(zip(acc,auc))
z=list(zip(numpy.arange(10),t))
d= {key: value for (key, value) in z}
df = pd.DataFrame(data=d,index=['accuracy score','auc score'])
df.to_excel(writer,sheet_name='w2v')
writer.save()


'''
###################################################################################################################################################################
naam = ["model_epoch2", "model_epoch10"]
epoch =[2, 10]
for x in range(2):
    model = Doc2Vec(dm =0, dbow_words=1, min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=8)
    model.build_vocab(corp)
    for ep in range(epoch[x]):
        log.info('Epoch %d' % ep)
        model.train(sentences.sentences_perm(),
                    total_examples=model.corpus_count,
                    epochs=model.iter,)
    model.save('./'+str(naam[x])+'.d2v')
    log.info('Model'+str(naam[x])+'Saved')


'''

skf = StratifiedKFold(n_splits=10)
classifier = LogisticRegression()

acc=numpy.zeros(10)

score=[numpy.zeros(10), numpy.zeros(10), numpy.zeros(10), numpy.zeros(10), numpy.zeros(10)]

#naam = ["model_epoch1", "model_epoch2", "model_epoch5", "model_epoch10", "model_epoch20"]
#naam = ["model_size020", "model_size050", "model_size100", "model_size300", "model_size700"]
#naam = ["model_neg1", "model_neg2", "model_neg5", "model_neg20", "model_neg100"]
naam = ["model_sub-1", "model_sub-2", "model_sub-3", "model_sub-4", "model_sub-5", "model_sub-6", "model_sub-7"]

writer = pd.ExcelWriter('hypercomp3.xlsx')

x = [numpy.zeros((50000,100)), numpy.zeros((50000,100)), numpy.zeros((50000,100)), numpy.zeros((50000,100)), numpy.zeros((50000,100)), numpy.zeros((50000,100)), numpy.zeros((50000,100))]
y = [numpy.zeros(50000), numpy.zeros(50000), numpy.zeros(50000), numpy.zeros(50000), numpy.zeros(50000), numpy.zeros(50000), numpy.zeros(50000)]

for j in range(len(naam)):
    model = Doc2Vec.load('./'+str(naam[j])+'.d2v')
    log.info('Sentiment')

    for i in range(12500):
        x[j][i] = model.docvecs['TEST_NEG_'+str(i)]
        x[j][12500+i] = model.docvecs['TRAIN_NEG_'+str(i)]
        x[j][25000+i] = model.docvecs['TEST_POS_'+str(i)]
        x[j][37500+i] = model.docvecs['TRAIN_POS_'+str(i)]
        y[j][i] = 0
        y[j][12500+i] = 0
        y[j][37500+i] = 1
        y[j][25000+i] = 1

    i = 0
    for train_index, test_index in skf.split(x[j],y[j]):
        print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = x[j][train_index], x[j][test_index]
        y_train, y_test = y[j][train_index], y[j][test_index]
        classifier.fit(x_train, y_train)
        acc[i] = classifier.score(x_test, y_test)
        i = i + 1

    z=list(zip(numpy.arange(10),acc))
    d= {key: value for (key, value) in z}
    df = pd.DataFrame(data=d,index=['accuracy score'])
    df.to_excel(writer,sheet_name=naam[j])

writer.save()
'''
    ##############################################################################################################################################################
    ##cv2
    for i in range(5000):
        test_arrays[1][i] = model.docvecs['TEST_POS_'+str(i+5000)]
        test_arrays[1][5000 + i] = model.docvecs['TEST_NEG_'+str(i+5000)]
        test_labels[1][i] = 1
        test_labels[1][5000 + i] = 0

    for i in range(5000):
        train_arrays[1][i] = model.docvecs['TEST_POS_'+str(i)]
        train_arrays[1][20000+i] = model.docvecs['TEST_NEG_'+str(i)]
        train_labels[1][i] = 1
        train_labels[1][20000+i] = 0
    for i in range(2500):
        train_arrays[1][5000+i] = model.docvecs['TEST_POS_'+str(10000+i)]
        train_arrays[1][25000 + i] = model.docvecs['TEST_NEG_'+str(10000+i)]
        train_labels[1][5000+i] = 1
        train_labels[1][25000 + i] = 0
    for i in range(12500):
        train_arrays[1][7500+i] = model.docvecs['TRAIN_POS_'+str(i)]
        train_arrays[1][27500 + i] = model.docvecs['TRAIN_NEG_'+str(i)]
        train_labels[1][7500+i] = 1
        train_labels[1][27500 + i] = 0

    #logistic regression classifier
    classifier = LogisticRegression()
    classifier.fit(train_arrays[1], train_labels[1])
    #Result
    log.info(classifier.score(test_arrays[1], test_labels[1]))
    score[x][1]=classifier.score(test_arrays[1], test_labels[1])
    ################################################################################################################################################################
    #CONFUSION MATRIX
    test_pred_labels = classifier.predict(test_arrays[1])
    test_pred_prob = classifier.predict_proba(test_arrays[1])[:, 1]
    ##CONFUSION MATRIX
    print("\nConfusion matrix:\n")
    confusion = metrics.confusion_matrix(test_labels[1], test_pred_labels)
    print(confusion)
    conf[x][1]=confusion

    ##cv3
    for i in range(2500):
        test_arrays[2][i] = model.docvecs['TEST_POS_'+str(i+10000)]
        test_arrays[2][5000 + i] = model.docvecs['TEST_NEG_'+str(i+10000)]
        test_labels[2][i] = 1
        test_labels[2][5000 + i] = 0
    for i in range(2500):
        test_arrays[2][2500+i] = model.docvecs['TRAIN_POS_'+str(i)]
        test_arrays[2][7500 + i] = model.docvecs['TRAIN_NEG_'+str(i)]
        test_labels[2][2500+i] = 1
        test_labels[2][7500 + i] = 0

    for i in range(10000):
        train_arrays[2][i] = model.docvecs['TEST_POS_'+str(i)]
        train_arrays[2][20000 + i] = model.docvecs['TEST_NEG_'+str(i)]
        train_labels[2][i] = 1
        train_labels[2][20000 + i] = 0

        train_arrays[2][10000+i] = model.docvecs['TRAIN_POS_'+str(2500+i)]
        train_arrays[2][30000 + i] = model.docvecs['TRAIN_NEG_'+str(2500+i)]
        train_labels[2][10000+i] = 1
        train_labels[2][30000 + i] = 0

    #logistic regression classifier
    classifier = LogisticRegression()
    classifier.fit(train_arrays[2], train_labels[2])
    #Result
    log.info(classifier.score(test_arrays[2], test_labels[2]))
    score[x][2]=classifier.score(test_arrays[2], test_labels[2])
    ################################################################################################################################################################
    #CONFUSION MATRIX
    test_pred_labels = classifier.predict(test_arrays[2])
    test_pred_prob = classifier.predict_proba(test_arrays[2])[:, 1]
    ##CONFUSION MATRIX
    print("\nConfusion matrix:\n")
    confusion = metrics.confusion_matrix(test_labels[2], test_pred_labels)
    print(confusion)
    conf[x][2]=confusion

    ##cv4
    for i in range(5000):
        test_arrays[3][i] = model.docvecs['TRAIN_POS_'+str(2500+i)]
        test_arrays[3][5000 + i] = model.docvecs['TRAIN_NEG_'+str(2500+i)]
        test_labels[3][i] = 1
        test_labels[3][5000 + i] = 0

    for i in range(12500):
        train_arrays[3][i] = model.docvecs['TEST_POS_'+str(i)]
        train_arrays[3][20000 + i] = model.docvecs['TEST_NEG_'+str(i)]
        train_labels[3][i] = 1
        train_labels[3][20000 + i] = 0
    for i in range(2500):
        train_arrays[3][12500 + i] = model.docvecs['TRAIN_POS_'+str(i)]
        train_arrays[3][32500 + i] = model.docvecs['TRAIN_NEG_'+str(i)]
        train_labels[3][12500 + i] = 1
        train_labels[3][32500 + i] = 0
    for i in range(5000):
        train_arrays[3][15000 + i] = model.docvecs['TRAIN_POS_'+str(7500+i)]
        train_arrays[3][35000 + i] = model.docvecs['TRAIN_NEG_'+str(7500+i)]
        train_labels[3][15000 + i] = 1
        train_labels[3][35000 + i] = 0

    #logistic regression classifier
    classifier = LogisticRegression()
    classifier.fit(train_arrays[3], train_labels[3])
    #Result
    log.info(classifier.score(test_arrays[3], test_labels[3]))
    score[x][3]=classifier.score(test_arrays[3], test_labels[3])
    ################################################################################################################################################################
    #CONFUSION MATRIX
    test_pred_labels = classifier.predict(test_arrays[3])
    test_pred_prob = classifier.predict_proba(test_arrays[3])[:, 1]
    ##CONFUSION MATRIX
    print("\nConfusion matrix:\n")
    confusion = metrics.confusion_matrix(test_labels[3], test_pred_labels)
    print(confusion)
    conf[x][3]=confusion

    ##cv5
    for i in range(5000):
        test_arrays[4][i] = model.docvecs['TRAIN_POS_'+str(7500+i)]
        test_arrays[4][5000 + i] = model.docvecs['TRAIN_NEG_'+str(7500+i)]
        test_labels[4][i] = 1
        test_labels[4][5000 + i] = 0

    for i in range(12500):
        train_arrays[4][i] = model.docvecs['TEST_POS_'+str(i)]
        train_arrays[4][20000 + i] = model.docvecs['TEST_NEG_'+str(i)]
        train_labels[4][i] = 1
        train_labels[4][20000 + i] = 0
    for i in range(7500):
        train_arrays[4][12500 + i] = model.docvecs['TRAIN_POS_'+str(i)]
        train_arrays[4][32500 + i] = model.docvecs['TRAIN_NEG_'+str(i)]
        train_labels[4][12500 + i] = 1
        train_labels[4][32500 + i] = 0

    #logistic regression classifier
    classifier = LogisticRegression()
    classifier.fit(train_arrays[4], train_labels[4])
    #Result
    log.info(classifier.score(test_arrays[4], test_labels[4]))
    score[x][4]=classifier.score(test_arrays[4], test_labels[4])
    ################################################################################################################################################################
    #CONFUSION MATRIX
    test_pred_labels = classifier.predict(test_arrays[4])
    test_pred_prob = classifier.predict_proba(test_arrays[4])[:, 1]
    ##CONFUSION MATRIX
    print("\nConfusion matrix:\n")
    confusion = metrics.confusion_matrix(test_labels[4], test_pred_labels)
    print(confusion)
    conf[x][4]=confusion

l=[]
for i in range(5):
    l.append(str(epoch[i]))
z=numpy.arange(5)

y1=[]
for i in range(5):
    y1.append(score[i][0])
y2=[]
for i in range(5):
    y2.append(score[i][1])
y3=[]
for i in range(5):
    y3.append(score[i][2])
y4=[]
for i in range(5):
    y4.append(score[i][3])
y5=[]
for i in range(5):
    y5.append(score[i][4])
ym=[]
for i in range(5):
    ym.append(max(score[i]))
yn=[]
for i in range(5):
    yn.append(min(score[i]))

plt.plot(z, y1,'m-',alpha = 0.4)
plt.plot(z, y2,'b-',alpha = 0.4)
plt.plot(z, y3,'g-',alpha = 0.4)
plt.plot(z, y4,'y-',alpha = 0.4)
plt.plot(z, y5,'c-',alpha = 0.4)
#plt.plot(z, ym,'r--d')
#plt.plot(z, yn,'r--d')

plt.title('accuracy vs epochs')
plt.xlabel('number of epochs')
plt.ylabel('accuracy score')
#plt.ylim(0.75,1)
plt.xticks(z,l)
plt.grid(True)

for i in range(5):
    plt.annotate("("+str(epoch[i])+","+str(y1[i])+")",xy=(z[i],y1[i]))
for i in range(5):
    plt.annotate("("+str(epoch[i])+","+str(y2[i])+")",xy=(z[i],y2[i]))
for i in range(5):
    plt.annotate("("+str(epoch[i])+","+str(y3[i])+")",xy=(z[i],y3[i]))
for i in range(5):
    plt.annotate("("+str(epoch[i])+","+str(y4[i])+")",xy=(z[i],y4[i]))
for i in range(5):
    plt.annotate("("+str(epoch[i])+","+str(y5[i])+")",xy=(z[i],y5[i]))

ymid=[]
for i in range(5):
    ymid.append(numpy.mean(score[i]))
yu = []
for i in range(5):
    yu.append(ymid[i]+2*numpy.std(score[i]))
yd = []
for i in range(5):
    yd.append(ymid[i]-2*numpy.std(score[i]))
plt.plot(z, ymid,'k-d')
plt.plot(z, yu,'r--d')
plt.plot(z, yd,'r--d')
plt.plot(z, y1,'m-',alpha = 0.4)
plt.plot(z, y2,'b-',alpha = 0.4)
plt.plot(z, y3,'g-',alpha = 0.4)
plt.plot(z, y4,'y-',alpha = 0.4)
plt.plot(z, y5,'c-',alpha = 0.4)
plt.title('accuracy vs epochs')
plt.xlabel('number of epochs')
plt.ylabel('accuracy score')
plt.ylim(0,1)
plt.xticks(z,l)
plt.grid(True)
for i in range(5):
    plt.annotate("("+str(epoch[i])+","+str(ymid[i])+")",xy=(z[i],ymid[i]))
for i in range(5):
    plt.annotate("("+str(epoch[i])+","+str(yu[i])+")",xy=(z[i],yu[i]))
for i in range(5):
    plt.annotate("("+str(epoch[i])+","+str(yd[i])+")",xy=(z[i],yd[i]))

'''

'''
t=list(zip(score,conf))
z=list(zip(naam,t))

d = {key: value for (key, value) in z}
df = pd.DataFrame(data=d,index=['accuracy score','confusion matrix'])

writer = pd.ExcelWriter('crossv.xlsx')
df.to_excel(writer,sheet_name='crossv',  merge_cells=True, startrow=0, startcol=0)
writer.save()
'''
'''