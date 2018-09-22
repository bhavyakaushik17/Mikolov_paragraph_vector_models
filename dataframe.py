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
'''
#######################################################################################################################################################
##Word2Vec
score=numpy.zeros(10)
c=numpy.zeros((2,2))
conf=[]
df=[]
for i in range(10):
    conf.append(c)

s=[100,300,700]
for x in range(len(s)):
    #corpus for word2vec
    w2v_corp= []
    for i in range(len(corp)):
        w2v_corp.append(corp[i][0])
    #training model
    model = Word2Vec(w2v_corp, min_count=1, size = s[x])
    #Saving the model

    #for loading the model
    #model1 = Word2Vec.load('./imdb.w2v')

    #doc vectors: v[i] ; averaging out word vectors for each document
    v=numpy.zeros((len(corp), s[x]))
    for i in range(len(w2v_corp)):
        for word in w2v_corp[i]:
            v[i]=v[i]+model[word]/len(w2v_corp[i])
    #end of training
    #############################################################################################################################################################
    #sentiment analysis word2vec
    log.info('Sentiment')

    #test set
    #The order of reading the input was noted, it will be used now to assign the polarity{0,1} of the reviews
    test_arrays_w2v = numpy.zeros((200, s[x]))
    test_labels_w2v = numpy.zeros(200)
    #v[0] to v[12499] is test_neg, v[12500] to v[24999] is test_pos
    for i in range(100):
        test_arrays_w2v[i] = v[i]
        test_arrays_w2v[100 + i] = v[100+i]
        test_labels_w2v[i] = 0
        test_labels_w2v[100 + i] = 1
    log.info(test_labels_w2v)

    #training set
    train_arrays_w2v = numpy.zeros((200, s[x]))
    train_labels_w2v = numpy.zeros(200)
    #v[25000] to v[37499] is train_neg, v[37500] to v[49999] is train_pos
    for i in range(100):
        train_arrays_w2v[i] = v[200+i]
        train_arrays_w2v[100 + i] = v[300+i]
        train_labels_w2v[i] = 0
        train_labels_w2v[100 + i] = 1
    log.info(train_labels_w2v)

    ################################################################################################################################################################
    #logistic regression classifier
    classifier=LogisticRegression()
    ###Using default parameters for logistic regression as follows:
    ###LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
    ###          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)
    log.info('Fitting')
    classifier.fit(v[200:400], train_labels_w2v)

    #result
    log.info(classifier.score(v[0:200], test_labels_w2v))
    score[x]=classifier.score(v[0:200], test_labels_w2v)
    test_pred_labels = classifier.predict(test_arrays_w2v)
    test_pred_prob = classifier.predict_proba(test_arrays_w2v)[:, 1]

    print("\nConfusion matrix:\n")
    confusion = metrics.confusion_matrix(test_labels_w2v, test_pred_labels)
    print(confusion)
    conf[x]=confusion

t=list(zip(score,conf))
z=list(zip(s,t))

d= {key: value for (key, value) in z}
df.append(pd.DataFrame(data=d,index=['accuracy score','confusion matrix']))
#######################################################################################################################################################################
s=[1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100]
for x in range(len(s)):
    #corpus for word2vec
    w2v_corp= []
    for i in range(len(corp)):
        w2v_corp.append(corp[i][0])
    #training model
    model = Word2Vec(w2v_corp, min_count=1, size=100,sample = s[x])
    #Saving the model

    #for loading the model
    #model1 = Word2Vec.load('./imdb.w2v')

    #doc vectors: v[i] ; averaging out word vectors for each document
    v=numpy.zeros((len(corp), 100))
    for i in range(len(w2v_corp)):
        for word in w2v_corp[i]:
            v[i]=v[i]+model[word]/len(w2v_corp[i])
    #end of training
    #############################################################################################################################################################
    #sentiment analysis word2vec
    log.info('Sentiment')
    test_arrays_w2v = numpy.zeros((200, 100))
    test_labels_w2v = numpy.zeros(200)
    for i in range(100):
        test_arrays_w2v[i] = v[i]
        test_arrays_w2v[100 + i] = v[100+i]
        test_labels_w2v[i] = 0
        test_labels_w2v[100 + i] = 1
    log.info(test_labels_w2v)

    #training set
    train_arrays_w2v = numpy.zeros((200, 100))
    train_labels_w2v = numpy.zeros(200)
    #v[25000] to v[37499] is train_neg, v[37500] to v[49999] is train_pos
    for i in range(100):
        train_arrays_w2v[i] = v[200+i]
        train_arrays_w2v[100 + i] = v[300+i]
        train_labels_w2v[i] = 0
        train_labels_w2v[100 + i] = 1
    log.info(train_labels_w2v)

    ################################################################################################################################################################
    #logistic regression classifier
    classifier=LogisticRegression()
    ###Using default parameters for logistic regression as follows:
    ###LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
    ###          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)
    log.info('Fitting')
    classifier.fit(v[200:400], train_labels_w2v)

    #result
    log.info(classifier.score(v[0:200], test_labels_w2v))
    score[x]=classifier.score(v[0:200], test_labels_w2v)
    test_pred_labels = classifier.predict(test_arrays_w2v)
    test_pred_prob = classifier.predict_proba(test_arrays_w2v)[:, 1]

    print("\nConfusion matrix:\n")
    confusion = metrics.confusion_matrix(test_labels_w2v, test_pred_labels)
    print(confusion)
    conf[x]=confusion

t=list(zip(score,conf))
z=list(zip(s,t))

d= {key: value for (key, value) in z}
df.append(pd.DataFrame(data=d,index=['accuracy score','confusion matrix']))
#######################################################################################################################################################################
s=[0,1,2,5,20,100]
for x in range(len(s)):
    #corpus for word2vec
    w2v_corp= []
    for i in range(len(corp)):
        w2v_corp.append(corp[i][0])
    #training model
    model = Word2Vec(w2v_corp, min_count=1, size=100,negative = s[x])
    #Saving the model

    #for loading the model
    #model1 = Word2Vec.load('./imdb.w2v')

    #doc vectors: v[i] ; averaging out word vectors for each document
    v=numpy.zeros((len(corp), 100))
    for i in range(len(w2v_corp)):
        for word in w2v_corp[i]:
            v[i]=v[i]+model[word]/len(w2v_corp[i])
    #end of training
    #############################################################################################################################################################
    #sentiment analysis word2vec
    log.info('Sentiment')

    #test set
    #The order of reading the input was noted, it will be used now to assign the polarity{0,1} of the reviews
    test_arrays_w2v = numpy.zeros((200, 100))
    test_labels_w2v = numpy.zeros(200)
    #v[0] to v[12499] is test_neg, v[12500] to v[24999] is test_pos
    for i in range(100):
        test_arrays_w2v[i] = v[i]
        test_arrays_w2v[100 + i] = v[100+i]
        test_labels_w2v[i] = 0
        test_labels_w2v[100 + i] = 1
    log.info(test_labels_w2v)

    #training set
    train_arrays_w2v = numpy.zeros((200, 100))
    train_labels_w2v = numpy.zeros(200)
    #v[25000] to v[37499] is train_neg, v[37500] to v[49999] is train_pos
    for i in range(100):
        train_arrays_w2v[i] = v[200+i]
        train_arrays_w2v[100 + i] = v[300+i]
        train_labels_w2v[i] = 0
        train_labels_w2v[100 + i] = 1
    log.info(train_labels_w2v)

    ################################################################################################################################################################
    #logistic regression classifier
    classifier=LogisticRegression()
    ###Using default parameters for logistic regression as follows:
    ###LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
    ###          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)
    log.info('Fitting')
    classifier.fit(v[200:400], train_labels_w2v)

    #result
    log.info(classifier.score(v[0:200], test_labels_w2v))
    score[x]=classifier.score(v[0:200], test_labels_w2v)
    test_pred_labels = classifier.predict(test_arrays_w2v)
    test_pred_prob = classifier.predict_proba(test_arrays_w2v)[:, 1]

    print("\nConfusion matrix:\n")
    confusion = metrics.confusion_matrix(test_labels_w2v, test_pred_labels)
    print(confusion)
    conf[x]=confusion

t=list(zip(score,conf))
z=list(zip(s,t))

d= {key: value for (key, value) in z}
df.append(pd.DataFrame(data=d,index=['accuracy score','confusion matrix']))
#########################################################################################################################################################
s=[1,5,20,100]
for x in range(len(s)):
    #corpus for word2vec
    w2v_corp= []
    for i in range(len(corp)):
        w2v_corp.append(corp[i][0])
    #training model
    model = Word2Vec(w2v_corp, min_count=1, size=100,iter = s[x])
    #Saving the model

    #for loading the model
    #model1 = Word2Vec.load('./imdb.w2v')

    #doc vectors: v[i] ; averaging out word vectors for each document
    v=numpy.zeros((len(corp), 100))
    for i in range(len(w2v_corp)):
        for word in w2v_corp[i]:
            v[i]=v[i]+model[word]/len(w2v_corp[i])
    #end of training
    #############################################################################################################################################################
    #sentiment analysis word2vec
    log.info('Sentiment')

    #test set
    #The order of reading the input was noted, it will be used now to assign the polarity{0,1} of the reviews
    test_arrays_w2v = numpy.zeros((200, 100))
    test_labels_w2v = numpy.zeros(200)
    #v[0] to v[12499] is test_neg, v[12500] to v[24999] is test_pos
    for i in range(100):
        test_arrays_w2v[i] = v[i]
        test_arrays_w2v[100 + i] = v[100+i]
        test_labels_w2v[i] = 0
        test_labels_w2v[100 + i] = 1
    log.info(test_labels_w2v)

    #training set
    train_arrays_w2v = numpy.zeros((200, 100))
    train_labels_w2v = numpy.zeros(200)
    #v[25000] to v[37499] is train_neg, v[37500] to v[49999] is train_pos
    for i in range(100):
        train_arrays_w2v[i] = v[200+i]
        train_arrays_w2v[100 + i] = v[300+i]
        train_labels_w2v[i] = 0
        train_labels_w2v[100 + i] = 1
    log.info(train_labels_w2v)

    ################################################################################################################################################################
    #logistic regression classifier
    classifier=LogisticRegression()
    ###Using default parameters for logistic regression as follows:
    ###LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
    ###          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)
    log.info('Fitting')
    classifier.fit(v[200:400], train_labels_w2v)

    #result
    log.info(classifier.score(v[0:200], test_labels_w2v))
    score[x]=classifier.score(v[0:200], test_labels_w2v)
    test_pred_labels = classifier.predict(test_arrays_w2v)
    test_pred_prob = classifier.predict_proba(test_arrays_w2v)[:, 1]

    print("\nConfusion matrix:\n")
    confusion = metrics.confusion_matrix(test_labels_w2v, test_pred_labels)
    print(confusion)
    conf[x]=confusion

t=list(zip(score,conf))
z=list(zip(s,t))

d= {key: value for (key, value) in z}
df.append(pd.DataFrame(data=d,index=['accuracy score','confusion matrix']))
####################################################################################################################################################################################


w2v=pd.concat(df,keys=['dimention of vector','random sampling','negative sampling','epochs'],axis=1)
writer = pd.ExcelWriter('output.xlsx')
w2v.to_excel(writer,sheet_name='w2v',  merge_cells=True, startrow=0, startcol=0)

#############################################################################################################################################################################################
vocab=list(model.wv.vocab.keys())

allwords=[]
for i in range(len(corp)):
    print(i)
    for word in corp[i][0]:
        allwords.append(word)
#########One hot not possible with vocabulary size 100,000 (datast : 10 million words 50,000 documents)
        #trying it with vocabulary size 10,000 (dataset: 100,000 words 400 documents)
one_hot=numpy.zeros(len(vocab))
vec=[]
for i in range(len(vocab)):
    print(i*100/len(vocab))
    vec.append(one_hot.copy())
for i in range(len(vocab)):
    vec[i][i]=1
#trained word vectors
#now document vectors

#function to return vector of word given word:
def onehotvec(word):
    for i in range(len(vocab)):
        if word == vocab[i]:
            return vec[i]

dvec=numpy.zeros((len(corp), len(vocab)))
for i in range(len(corp)):
    print(i)
    for word in w2v_corp[i]:
        dvec[i]=dvec[i]+onehotvec(word)/len(w2v_corp[i])
# trained document vector
##################################################################################################################
# now sentiment
#sentiment analysis word2vec
log.info('Sentiment')

#test set
#The order of reading the input was noted, it will be used now to assign the polarity{0,1} of the reviews
test_arrays_onehot = numpy.zeros((200, len(vocab)))
test_labels_onehot = numpy.zeros(200)
for i in range(100):
    test_arrays_onehot[i] = dvec[i]
    test_arrays_onehot[100 + i] = dvec[100+i]
    test_labels_onehot[i] = 0
    test_labels_onehot[100 + i] = 1
log.info(test_labels_onehot)

#training set
train_arrays_onehot = numpy.zeros((200, len(vocab)))
train_labels_onehot = numpy.zeros(200)
for i in range(100):
    train_arrays_onehot[i] = dvec[200+i]
    train_arrays_onehot[100 + i] = dvec[300+i]
    train_labels_onehot[i] = 0
    train_labels_onehot[100 + i] = 1
log.info(train_labels_onehot)
#########################################################################################################################
log.info('Fitting')
classifier = LogisticRegression()
classifier.fit(train_arrays_onehot, train_labels_onehot)
#Result
log.info(classifier.score(test_arrays_onehot, test_labels_onehot))
#################################################################################################################################################################
###Confusion matrix

#predicted labels and predicted probabilities
test_pred_labels_onehot = classifier.predict(test_arrays_onehot)
test_pred_prob_onehot = classifier.predict_proba(test_arrays_onehot)[:, 1]

##CONFUSION MATRIX
print("\nConfusion matrix:\n")
confusion = metrics.confusion_matrix(test_labels_onehot, test_pred_labels_onehot)
print(confusion)

bow=pd.DataFrame(data=[classifier.score(test_arrays_onehot, test_labels_onehot),confusion],index=['accuracy score','confusion matrix'])
bow.to_excel(writer,sheet_name='bow',  merge_cells=True, startrow=0, startcol=0,header=False)
###################################################################################################################################
# at the end:
writer.save()

#########################################################
x=list(df[0].columns)
y=list(df[0].values[0])

plt.plot(x, y)
plt.title('accuracy vs dim')
plt.xlabel('dimention of vector')
plt.ylabel('accuracy score')
plt.ylim([0.0, 1.0])
plt.grid(True)
for i in range(len(x)):
    plt.annotate("("+str(x[i])+","+str(y[i])+")",xy=(x[i],y[i]))
####################################################

x=list(df[1].columns)
y=list(df[1].values[0])

plt.plot(x, y,'b-d')
plt.title('accuracy vs sub-sampling rate')
plt.xlabel('sub-sampling rate')
plt.ylabel('accuracy score')
plt.ylim([0.0, 1.0])
plt.grid(True)
for i in range(len(x)):
    plt.annotate("("+str(x[i])+","+str(y[i])+")",xy=(x[i],y[i]))
############################################
x=list(df[2].columns)
y=list(df[2].values[0])

plt.plot(x, y)
plt.title('accuracy vs negative sampling rate')
plt.xlabel('negative sampling rate')
plt.ylabel('accuracy score')
plt.ylim([0.0, 1.0])
plt.grid(True)
for i in range(len(x)):
    plt.annotate("("+str(x[i])+","+str(y[i])+")",xy=(x[i],y[i]))
########################################################################
x=list(df[3].columns)
y=list(df[3].values[0])

plt.plot(x, y)
plt.title('accuracy vs no. of epochs')
plt.xlabel('epochs')
plt.ylabel('accuracy score')
plt.ylim([0.0, 1.0])
plt.grid(True)
for i in range(len(x)):
    plt.annotate("("+str(x[i])+","+str(y[i])+")",xy=(x[i],y[i]))
#########################################################################################
#clear all
#%reset
####################################################
'''
score=numpy.zeros(3)
c=numpy.zeros((2,2))
conf=[]
df=[]
for i in range(3):
    conf.append(c)

s=[100, 300, 700]
naam = ["model_size100","model_size300","model_size700"]
rsize=[]
for x in range(len(s)):
    tic=time.clock()
    model = Doc2Vec(dm =0, dbow_words=1, min_count=1, window=10, size=s[x], sample=1e-4, negative=5, workers=8)
    model.build_vocab(corp)
    for epoch in range(1):
        log.info('Epoch %d' % epoch)
        model.train(sentences.sentences_perm(),
                    total_examples=model.corpus_count,
                    epochs=model.iter,)

    toc=time.clock()
    rsize.append(toc-tic)
    model.save('./'+str(naam[x])+'.d2v')
    log.info('Model'+str(naam[x])+'Saved')

s=[1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1]
rsub=[]
naam = ["model_sub-7","model_sub-6","model_sub-5","model_sub-4","model_sub-3","model_sub-2","model_sub-1","model_sub0"]
for x in range(len(s)):
    tic=time.clock()
    model = Doc2Vec(dm =0, dbow_words=1, min_count=1, window=10, size=100, sample=s[x], negative=5, workers=8)
    model.build_vocab(corp)
    for epoch in range(1):
        log.info('Epoch %d' % epoch)
        model.train(sentences.sentences_perm(),
                    total_examples=model.corpus_count,
                    epochs=model.iter,)

    toc=time.clock()
    rsub.append(toc-tic)
    model.save('./'+str(naam[x])+'.d2v')
    log.info('Model'+str(naam[x])+'Saved')

s=[0, 2, 5, 20, 100]
rneg=[]
naam = ["model_neg0","model_neg2","model_neg5","model_neg20","model_neg100"]
for x in range(len(s)):
    tic=time.clock()
    model = Doc2Vec(dm =0, dbow_words=1, min_count=1, window=10, size=100, sample=1e-4, negative=s[x], workers=8)
    model.build_vocab(corp)
    for epoch in range(1):
        log.info('Epoch %d' % epoch)
        model.train(sentences.sentences_perm(),
                    total_examples=model.corpus_count,
                    epochs=model.iter,)

    toc=time.clock()
    rneg.append(toc-tic)
    model.save('./'+str(naam[x])+'.d2v')
    log.info('Model'+str(naam[x])+'Saved')

s=[0, 1, 5, 20]
repoch=[]
naam = ["model_epoch0","model_epoch1","model_epoch5","model_epoch20"]
for x in range(len(s)):
    tic=time.clock()
    model = Doc2Vec(dm =0, dbow_words=1, min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=8)
    model.build_vocab(corp)
    for epoch in range(s[x]):
        log.info('Epoch %d' % epoch)
        model.train(sentences.sentences_perm(),
                    total_examples=model.corpus_count,
                    epochs=model.iter,)

    toc=time.clock()
    repoch.append(toc-tic)
    model.save('./'+str(naam[x])+'.d2v')
    log.info('Model'+str(naam[x])+'Saved')

#end of training
#############################################################################################################################################################
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

################################################################################################################################################################
#logistic regression classifier
log.info('Fitting')
classifier = LogisticRegression()
classifier.fit(train_arrays_d2v, train_labels_d2v)
#Result
log.info(classifier.score(test_arrays_d2v, test_labels_d2v))
    ################################################################################################################################################################
    #CONFUSION MATRIX
test_pred_labels_d2v = classifier.predict(test_arrays_d2v)
test_pred_prob_d2v = classifier.predict_proba(test_arrays_d2v)[:, 1]
##CONFUSION MATRIX
print("\nConfusion matrix:\n")
confusion = metrics.confusion_matrix(test_labels_d2v, test_pred_labels_d2v)
print(confusion)


    score[x]=classifier.score(test_arrays_d2v, test_labels_d2v)
    conf[x]=confusion

t=list(zip(score,conf))
z=list(zip(s,t))

d= {key: value for (key, value) in z}
df.append(pd.DataFrame(data=d,index=['accuracy score','confusion matrix']))
##############################################################################################################################################
s=[1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100]
for x in range(len(s)):
    model = Doc2Vec(dm =1, dbow_words=0, min_count=1, window=10, size=100, sample=s[x], negative=5, workers=8)
    model.build_vocab(corp)
    for epoch in range(5):
        log.info('Epoch %d' % epoch)
        model.train(sentences.sentences_perm(),
                    total_examples=model.corpus_count,
                    epochs=model.iter,)

    #end of training
    #############################################################################################################################################################
    log.info('Sentiment')

    train_arrays_d2v = numpy.zeros((200, 100))
    train_labels_d2v = numpy.zeros(200)
    for i in range(100):
        prefix_train_pos = 'TRAIN_POS_' + str(i)
        prefix_train_neg = 'TRAIN_NEG_' + str(i)
        train_arrays_d2v[i] = model.docvecs[prefix_train_pos]
        train_arrays_d2v[100 + i] = model.docvecs[prefix_train_neg]
        train_labels_d2v[i] = 1
        train_labels_d2v[100 + i] = 0
    log.info(train_labels_d2v)

    test_arrays_d2v = numpy.zeros((200, 100))
    test_labels_d2v = numpy.zeros(200)
    for i in range(100):
        prefix_test_pos = 'TEST_POS_' + str(i)
        prefix_test_neg = 'TEST_NEG_' + str(i)
        test_arrays_d2v[i] = model.docvecs[prefix_test_pos]
        test_arrays_d2v[100 + i] = model.docvecs[prefix_test_neg]
        test_labels_d2v[i] = 1
        test_labels_d2v[100 + i] = 0
    log.info(test_labels_d2v)

    ################################################################################################################################################################
    #logistic regression classifier
    log.info('Fitting')
    classifier = LogisticRegression()
    classifier.fit(train_arrays_d2v, train_labels_d2v)
    #Result
    log.info(classifier.score(test_arrays_d2v, test_labels_d2v))
    ################################################################################################################################################################
    #CONFUSION MATRIX
    test_pred_labels_d2v = classifier.predict(test_arrays_d2v)
    test_pred_prob_d2v = classifier.predict_proba(test_arrays_d2v)[:, 1]
    ##CONFUSION MATRIX
    print("\nConfusion matrix:\n")
    confusion = metrics.confusion_matrix(test_labels_d2v, test_pred_labels_d2v)
    print(confusion)
    score[x]=classifier.score(test_arrays_d2v, test_labels_d2v)
    conf[x]=confusion

t=list(zip(score,conf))
z=list(zip(s,t))

d= {key: value for (key, value) in z}
df.append(pd.DataFrame(data=d,index=['accuracy score','confusion matrix']))
###############################################################################################################################################
s=[0,1,2,5,20,100]
for x in range(len(s)):
    model = Doc2Vec(dm =1, dbow_words=0, min_count=1, window=10, size=100, sample=1e-4, negative=s[x], workers=8)
    model.build_vocab(corp)
    for epoch in range(5):
        log.info('Epoch %d' % epoch)
        model.train(sentences.sentences_perm(),
                    total_examples=model.corpus_count,
                    epochs=model.iter,)

    #end of training
    #############################################################################################################################################################
    log.info('Sentiment')

    train_arrays_d2v = numpy.zeros((200, 100))
    train_labels_d2v = numpy.zeros(200)
    for i in range(100):
        prefix_train_pos = 'TRAIN_POS_' + str(i)
        prefix_train_neg = 'TRAIN_NEG_' + str(i)
        train_arrays_d2v[i] = model.docvecs[prefix_train_pos]
        train_arrays_d2v[100 + i] = model.docvecs[prefix_train_neg]
        train_labels_d2v[i] = 1
        train_labels_d2v[100 + i] = 0
    log.info(train_labels_d2v)

    test_arrays_d2v = numpy.zeros((200, 100))
    test_labels_d2v = numpy.zeros(200)
    for i in range(100):
        prefix_test_pos = 'TEST_POS_' + str(i)
        prefix_test_neg = 'TEST_NEG_' + str(i)
        test_arrays_d2v[i] = model.docvecs[prefix_test_pos]
        test_arrays_d2v[100 + i] = model.docvecs[prefix_test_neg]
        test_labels_d2v[i] = 1
        test_labels_d2v[100 + i] = 0
    log.info(test_labels_d2v)

    ################################################################################################################################################################
    #logistic regression classifier
    log.info('Fitting')
    classifier = LogisticRegression()
    classifier.fit(train_arrays_d2v, train_labels_d2v)
    #Result
    log.info(classifier.score(test_arrays_d2v, test_labels_d2v))
    ################################################################################################################################################################
    #CONFUSION MATRIX
    test_pred_labels_d2v = classifier.predict(test_arrays_d2v)
    test_pred_prob_d2v = classifier.predict_proba(test_arrays_d2v)[:, 1]
    ##CONFUSION MATRIX
    print("\nConfusion matrix:\n")
    confusion = metrics.confusion_matrix(test_labels_d2v, test_pred_labels_d2v)
    print(confusion)
    score[x]=classifier.score(test_arrays_d2v, test_labels_d2v)
    conf[x]=confusion

t=list(zip(score,conf))
z=list(zip(s,t))

d= {key: value for (key, value) in z}
df.append(pd.DataFrame(data=d,index=['accuracy score','confusion matrix']))
###################################################################################################################################################################
s=[1,5,20,100]
for x in range(len(s)):
    model = Doc2Vec(dm =1, dbow_words=0, min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=8)
    model.build_vocab(corp)
    for epoch in range(s[x]):
        log.info('Epoch %d' % epoch)
        model.train(sentences.sentences_perm(),
                    total_examples=model.corpus_count,
                    epochs=model.iter,)

    #end of training
    #############################################################################################################################################################
    log.info('Sentiment')

    train_arrays_d2v = numpy.zeros((200, 100))
    train_labels_d2v = numpy.zeros(200)
    for i in range(100):
        prefix_train_pos = 'TRAIN_POS_' + str(i)
        prefix_train_neg = 'TRAIN_NEG_' + str(i)
        train_arrays_d2v[i] = model.docvecs[prefix_train_pos]
        train_arrays_d2v[100 + i] = model.docvecs[prefix_train_neg]
        train_labels_d2v[i] = 1
        train_labels_d2v[100 + i] = 0
    log.info(train_labels_d2v)

    test_arrays_d2v = numpy.zeros((200, 100))
    test_labels_d2v = numpy.zeros(200)
    for i in range(100):
        prefix_test_pos = 'TEST_POS_' + str(i)
        prefix_test_neg = 'TEST_NEG_' + str(i)
        test_arrays_d2v[i] = model.docvecs[prefix_test_pos]
        test_arrays_d2v[100 + i] = model.docvecs[prefix_test_neg]
        test_labels_d2v[i] = 1
        test_labels_d2v[100 + i] = 0
    log.info(test_labels_d2v)

    ################################################################################################################################################################
    #logistic regression classifier
    log.info('Fitting')
    classifier = LogisticRegression()
    classifier.fit(train_arrays_d2v, train_labels_d2v)
    #Result
    log.info(classifier.score(test_arrays_d2v, test_labels_d2v))
    ################################################################################################################################################################
    #CONFUSION MATRIX
    test_pred_labels_d2v = classifier.predict(test_arrays_d2v)
    test_pred_prob_d2v = classifier.predict_proba(test_arrays_d2v)[:, 1]
    ##CONFUSION MATRIX
    print("\nConfusion matrix:\n")
    confusion = metrics.confusion_matrix(test_labels_d2v, test_pred_labels_d2v)
    print(confusion)
    score[x]=classifier.score(test_arrays_d2v, test_labels_d2v)
    conf[x]=confusion

t=list(zip(score,conf))
z=list(zip(s,t))

d= {key: value for (key, value) in z}
df.append(pd.DataFrame(data=d,index=['accuracy score','confusion matrix']))
#################################################################################################################################################################
pvdm=pd.concat(df,keys=['dimention of vector','random sampling','negative sampling','epochs'],axis=1)
writer = pd.ExcelWriter('output.xlsx')
pvdm.to_excel(writer,sheet_name='pvdm',  merge_cells=True, startrow=0, startcol=0)

##################################################################################################################################################################
#########################################################
x=list(df[0].columns)
y=list(df[0].values[0])

plt.plot(x, y)
plt.title('accuracy vs dim')
plt.xlabel('dimention of vector')
plt.ylabel('accuracy score')
plt.ylim([0.0, 1.0])
plt.grid(True)
for i in range(len(x)):
    plt.annotate("("+str(x[i])+","+str(y[i])+")",xy=(x[i],y[i]))
####################################################

x=list(df[1].columns)
y=list(df[1].values[0])

plt.plot(x, y,'b-d')
plt.title('accuracy vs sub-sampling rate')
plt.xlabel('sub-sampling rate')
plt.ylabel('accuracy score')
plt.ylim([0.0, 1.0])
plt.grid(True)
for i in range(len(x)):
    plt.annotate("("+str(x[i])+","+str(y[i])+")",xy=(x[i],y[i]))
############################################
x=list(df[2].columns)
y=list(df[2].values[0])

plt.plot(x, y)
plt.title('accuracy vs negative sampling rate')
plt.xlabel('negative sampling rate')
plt.ylabel('accuracy score')
plt.ylim([0.0, 1.0])
plt.grid(True)
for i in range(len(x)):
    plt.annotate("("+str(x[i])+","+str(y[i])+")",xy=(x[i],y[i]))
########################################################################
x=list(df[3].columns)
y=list(df[3].values[0])

plt.plot(x, y)
plt.title('accuracy vs no. of epochs')
plt.xlabel('epochs')
plt.ylabel('accuracy score')
plt.ylim([0.0, 1.0])
plt.grid(True)
for i in range(len(x)):
    plt.annotate("("+str(x[i])+","+str(y[i])+")",xy=(x[i],y[i]))
#########################################################################################
#clear all
#%reset
################################################################################################################
score=numpy.zeros(10)
c=numpy.zeros((2,2))
conf=[]
df1=[]
for i in range(10):
    conf.append(c)

s=[100,200,300,400,500,600,700,800,900,1000]
for x in range(len(s)):
    model = Doc2Vec(dm =0, dbow_words=0, min_count=1, window=10, size=s[x], sample=1e-4, negative=5, workers=8)
    model.build_vocab(corp)
    for epoch in range(5):
        log.info('Epoch %d' % epoch)
        model.train(sentences.sentences_perm(),
                    total_examples=model.corpus_count,
                    epochs=model.iter,)

    #end of training
    #############################################################################################################################################################
    log.info('Sentiment')

    train_arrays_d2v = numpy.zeros((200, s[x]))
    train_labels_d2v = numpy.zeros(200)
    for i in range(100):
        prefix_train_pos = 'TRAIN_POS_' + str(i)
        prefix_train_neg = 'TRAIN_NEG_' + str(i)
        train_arrays_d2v[i] = model.docvecs[prefix_train_pos]
        train_arrays_d2v[100 + i] = model.docvecs[prefix_train_neg]
        train_labels_d2v[i] = 1
        train_labels_d2v[100 + i] = 0
    log.info(train_labels_d2v)

    test_arrays_d2v = numpy.zeros((200, s[x]))
    test_labels_d2v = numpy.zeros(200)
    for i in range(100):
        prefix_test_pos = 'TEST_POS_' + str(i)
        prefix_test_neg = 'TEST_NEG_' + str(i)
        test_arrays_d2v[i] = model.docvecs[prefix_test_pos]
        test_arrays_d2v[100 + i] = model.docvecs[prefix_test_neg]
        test_labels_d2v[i] = 1
        test_labels_d2v[100 + i] = 0
    log.info(test_labels_d2v)

    ################################################################################################################################################################
    #logistic regression classifier
    log.info('Fitting')
    classifier = LogisticRegression()
    classifier.fit(train_arrays_d2v, train_labels_d2v)
    #Result
    log.info(classifier.score(test_arrays_d2v, test_labels_d2v))
    ################################################################################################################################################################
    #CONFUSION MATRIX
    test_pred_labels_d2v = classifier.predict(test_arrays_d2v)
    test_pred_prob_d2v = classifier.predict_proba(test_arrays_d2v)[:, 1]
    ##CONFUSION MATRIX
    print("\nConfusion matrix:\n")
    confusion = metrics.confusion_matrix(test_labels_d2v, test_pred_labels_d2v)
    print(confusion)
    score[x]=classifier.score(test_arrays_d2v, test_labels_d2v)
    conf[x]=confusion

t=list(zip(score,conf))
z=list(zip(s,t))

d= {key: value for (key, value) in z}
df1.append(pd.DataFrame(data=d,index=['accuracy score','confusion matrix']))
##############################################################################################################################################
s=[1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100]
for x in range(len(s)):
    model = Doc2Vec(dm =0, dbow_words=0, min_count=1, window=10, size=100, sample=s[x], negative=5, workers=8)
    model.build_vocab(corp)
    for epoch in range(5):
        log.info('Epoch %d' % epoch)
        model.train(sentences.sentences_perm(),
                    total_examples=model.corpus_count,
                    epochs=model.iter,)

    #end of training
    #############################################################################################################################################################
    log.info('Sentiment')

    train_arrays_d2v = numpy.zeros((200, 100))
    train_labels_d2v = numpy.zeros(200)
    for i in range(100):
        prefix_train_pos = 'TRAIN_POS_' + str(i)
        prefix_train_neg = 'TRAIN_NEG_' + str(i)
        train_arrays_d2v[i] = model.docvecs[prefix_train_pos]
        train_arrays_d2v[100 + i] = model.docvecs[prefix_train_neg]
        train_labels_d2v[i] = 1
        train_labels_d2v[100 + i] = 0
    log.info(train_labels_d2v)

    test_arrays_d2v = numpy.zeros((200, 100))
    test_labels_d2v = numpy.zeros(200)
    for i in range(100):
        prefix_test_pos = 'TEST_POS_' + str(i)
        prefix_test_neg = 'TEST_NEG_' + str(i)
        test_arrays_d2v[i] = model.docvecs[prefix_test_pos]
        test_arrays_d2v[100 + i] = model.docvecs[prefix_test_neg]
        test_labels_d2v[i] = 1
        test_labels_d2v[100 + i] = 0
    log.info(test_labels_d2v)

    ################################################################################################################################################################
    #logistic regression classifier
    log.info('Fitting')
    classifier = LogisticRegression()
    classifier.fit(train_arrays_d2v, train_labels_d2v)
    #Result
    log.info(classifier.score(test_arrays_d2v, test_labels_d2v))
    ################################################################################################################################################################
    #CONFUSION MATRIX
    test_pred_labels_d2v = classifier.predict(test_arrays_d2v)
    test_pred_prob_d2v = classifier.predict_proba(test_arrays_d2v)[:, 1]
    ##CONFUSION MATRIX
    print("\nConfusion matrix:\n")
    confusion = metrics.confusion_matrix(test_labels_d2v, test_pred_labels_d2v)
    print(confusion)
    score[x]=classifier.score(test_arrays_d2v, test_labels_d2v)
    conf[x]=confusion

t=list(zip(score,conf))
z=list(zip(s,t))

d= {key: value for (key, value) in z}
df1.append(pd.DataFrame(data=d,index=['accuracy score','confusion matrix']))
###############################################################################################################################################
s=[0,1,2,5,20,100]
for x in range(len(s)):
    model = Doc2Vec(dm =0, dbow_words=0, min_count=1, window=10, size=100, sample=1e-4, negative=s[x], workers=8)
    model.build_vocab(corp)
    for epoch in range(5):
        log.info('Epoch %d' % epoch)
        model.train(sentences.sentences_perm(),
                    total_examples=model.corpus_count,
                    epochs=model.iter,)

    #end of training
    #############################################################################################################################################################
    log.info('Sentiment')

    train_arrays_d2v = numpy.zeros((200, 100))
    train_labels_d2v = numpy.zeros(200)
    for i in range(100):
        prefix_train_pos = 'TRAIN_POS_' + str(i)
        prefix_train_neg = 'TRAIN_NEG_' + str(i)
        train_arrays_d2v[i] = model.docvecs[prefix_train_pos]
        train_arrays_d2v[100 + i] = model.docvecs[prefix_train_neg]
        train_labels_d2v[i] = 1
        train_labels_d2v[100 + i] = 0
    log.info(train_labels_d2v)

    test_arrays_d2v = numpy.zeros((200, 100))
    test_labels_d2v = numpy.zeros(200)
    for i in range(100):
        prefix_test_pos = 'TEST_POS_' + str(i)
        prefix_test_neg = 'TEST_NEG_' + str(i)
        test_arrays_d2v[i] = model.docvecs[prefix_test_pos]
        test_arrays_d2v[100 + i] = model.docvecs[prefix_test_neg]
        test_labels_d2v[i] = 1
        test_labels_d2v[100 + i] = 0
    log.info(test_labels_d2v)

    ################################################################################################################################################################
    #logistic regression classifier
    log.info('Fitting')
    classifier = LogisticRegression()
    classifier.fit(train_arrays_d2v, train_labels_d2v)
    #Result
    log.info(classifier.score(test_arrays_d2v, test_labels_d2v))
    ################################################################################################################################################################
    #CONFUSION MATRIX
    test_pred_labels_d2v = classifier.predict(test_arrays_d2v)
    test_pred_prob_d2v = classifier.predict_proba(test_arrays_d2v)[:, 1]
    ##CONFUSION MATRIX
    print("\nConfusion matrix:\n")
    confusion = metrics.confusion_matrix(test_labels_d2v, test_pred_labels_d2v)
    print(confusion)
    score[x]=classifier.score(test_arrays_d2v, test_labels_d2v)
    conf[x]=confusion

t=list(zip(score,conf))
z=list(zip(s,t))

d= {key: value for (key, value) in z}
df1.append(pd.DataFrame(data=d,index=['accuracy score','confusion matrix']))
###################################################################################################################################################################
s=[1,5,20,100]
for x in range(len(s)):
    model = Doc2Vec(dm =0, dbow_words=0, min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=8)
    model.build_vocab(corp)
    for epoch in range(s[x]):
        log.info('Epoch %d' % epoch)
        model.train(sentences.sentences_perm(),
                    total_examples=model.corpus_count,
                    epochs=model.iter,)

    #end of training
    #############################################################################################################################################################
    log.info('Sentiment')

    train_arrays_d2v = numpy.zeros((200, 100))
    train_labels_d2v = numpy.zeros(200)
    for i in range(100):
        prefix_train_pos = 'TRAIN_POS_' + str(i)
        prefix_train_neg = 'TRAIN_NEG_' + str(i)
        train_arrays_d2v[i] = model.docvecs[prefix_train_pos]
        train_arrays_d2v[100 + i] = model.docvecs[prefix_train_neg]
        train_labels_d2v[i] = 1
        train_labels_d2v[100 + i] = 0
    log.info(train_labels_d2v)

    test_arrays_d2v = numpy.zeros((200, 100))
    test_labels_d2v = numpy.zeros(200)
    for i in range(100):
        prefix_test_pos = 'TEST_POS_' + str(i)
        prefix_test_neg = 'TEST_NEG_' + str(i)
        test_arrays_d2v[i] = model.docvecs[prefix_test_pos]
        test_arrays_d2v[100 + i] = model.docvecs[prefix_test_neg]
        test_labels_d2v[i] = 1
        test_labels_d2v[100 + i] = 0
    log.info(test_labels_d2v)

    ################################################################################################################################################################
    #logistic regression classifier
    log.info('Fitting')
    classifier = LogisticRegression()
    classifier.fit(train_arrays_d2v, train_labels_d2v)
    #Result
    log.info(classifier.score(test_arrays_d2v, test_labels_d2v))
    ################################################################################################################################################################
    #CONFUSION MATRIX
    test_pred_labels_d2v = classifier.predict(test_arrays_d2v)
    test_pred_prob_d2v = classifier.predict_proba(test_arrays_d2v)[:, 1]
    ##CONFUSION MATRIX
    print("\nConfusion matrix:\n")
    confusion = metrics.confusion_matrix(test_labels_d2v, test_pred_labels_d2v)
    print(confusion)
    score[x]=classifier.score(test_arrays_d2v, test_labels_d2v)
    conf[x]=confusion

t=list(zip(score,conf))
z=list(zip(s,t))

d= {key: value for (key, value) in z}
df1.append(pd.DataFrame(data=d,index=['accuracy score','confusion matrix']))
#################################################################################################################################################################
pvdbow_0=pd.concat(df1,keys=['dimention of vector','random sampling','negative sampling','epochs'],axis=1)
pvdbow_0.to_excel(writer,sheet_name='pvdbow_0',  merge_cells=True, startrow=0, startcol=0)

##################################################################################################################################################################
#########################################################
x=list(df1[0].columns)
y=list(df1[0].values[0])

plt.plot(x, y)
plt.title('accuracy vs dim')
plt.xlabel('dimention of vector')
plt.ylabel('accuracy score')
plt.ylim([0.0, 1.0])
plt.grid(True)
for i in range(len(x)):
    plt.annotate("("+str(x[i])+","+str(y[i])+")",xy=(x[i],y[i]))
####################################################

x=list(df1[1].columns)
y=list(df1[1].values[0])

plt.plot(x, y,'b-d')
plt.title('accuracy vs sub-sampling rate')
plt.xlabel('sub-sampling rate')
plt.ylabel('accuracy score')
plt.ylim([0.0, 1.0])
plt.grid(True)
for i in range(len(x)):
    plt.annotate("("+str(x[i])+","+str(y[i])+")",xy=(x[i],y[i]))
############################################
x=list(df1[2].columns)
y=list(df1[2].values[0])

plt.plot(x, y)
plt.title('accuracy vs negative sampling rate')
plt.xlabel('negative sampling rate')
plt.ylabel('accuracy score')
plt.ylim([0.0, 1.0])
plt.grid(True)
for i in range(len(x)):
    plt.annotate("("+str(x[i])+","+str(y[i])+")",xy=(x[i],y[i]))
########################################################################
x=list(df1[3].columns)
y=list(df1[3].values[0])

plt.plot(x, y)
plt.title('accuracy vs no. of epochs')
plt.xlabel('epochs')
plt.ylabel('accuracy score')
plt.ylim([0.0, 1.0])
plt.grid(True)
for i in range(len(x)):
    plt.annotate("("+str(x[i])+","+str(y[i])+")",xy=(x[i],y[i]))
###########################################################################################################################
score=numpy.zeros(10)
c=numpy.zeros((2,2))
conf=[]
df2=[]
for i in range(10):
    conf.append(c)

s=[100,200,300,400,500,600,700,800,900,1000]
for x in range(len(s)):
    model = Doc2Vec(dm =0, dbow_words=1, min_count=1, window=10, size=s[x], sample=1e-4, negative=5, workers=8)
    model.build_vocab(corp)
    for epoch in range(5):
        log.info('Epoch %d' % epoch)
        model.train(sentences.sentences_perm(),
                    total_examples=model.corpus_count,
                    epochs=model.iter,)

    #end of training
    #############################################################################################################################################################
    log.info('Sentiment')

    train_arrays_d2v = numpy.zeros((200, s[x]))
    train_labels_d2v = numpy.zeros(200)
    for i in range(100):
        prefix_train_pos = 'TRAIN_POS_' + str(i)
        prefix_train_neg = 'TRAIN_NEG_' + str(i)
        train_arrays_d2v[i] = model.docvecs[prefix_train_pos]
        train_arrays_d2v[100 + i] = model.docvecs[prefix_train_neg]
        train_labels_d2v[i] = 1
        train_labels_d2v[100 + i] = 0
    log.info(train_labels_d2v)

    test_arrays_d2v = numpy.zeros((200, s[x]))
    test_labels_d2v = numpy.zeros(200)
    for i in range(100):
        prefix_test_pos = 'TEST_POS_' + str(i)
        prefix_test_neg = 'TEST_NEG_' + str(i)
        test_arrays_d2v[i] = model.docvecs[prefix_test_pos]
        test_arrays_d2v[100 + i] = model.docvecs[prefix_test_neg]
        test_labels_d2v[i] = 1
        test_labels_d2v[100 + i] = 0
    log.info(test_labels_d2v)

    ################################################################################################################################################################
    #logistic regression classifier
    log.info('Fitting')
    classifier = LogisticRegression()
    classifier.fit(train_arrays_d2v, train_labels_d2v)
    #Result
    log.info(classifier.score(test_arrays_d2v, test_labels_d2v))
    ################################################################################################################################################################
    #CONFUSION MATRIX
    test_pred_labels_d2v = classifier.predict(test_arrays_d2v)
    test_pred_prob_d2v = classifier.predict_proba(test_arrays_d2v)[:, 1]
    ##CONFUSION MATRIX
    print("\nConfusion matrix:\n")
    confusion = metrics.confusion_matrix(test_labels_d2v, test_pred_labels_d2v)
    print(confusion)
    score[x]=classifier.score(test_arrays_d2v, test_labels_d2v)
    conf[x]=confusion

t=list(zip(score,conf))
z=list(zip(s,t))

d= {key: value for (key, value) in z}
df2.append(pd.DataFrame(data=d,index=['accuracy score','confusion matrix']))
##############################################################################################################################################
s=[1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100]
for x in range(len(s)):
    model = Doc2Vec(dm =0, dbow_words=0, min_count=1, window=10, size=100, sample=s[x], negative=5, workers=8)
    model.build_vocab(corp)
    for epoch in range(5):
        log.info('Epoch %d' % epoch)
        model.train(sentences.sentences_perm(),
                    total_examples=model.corpus_count,
                    epochs=model.iter,)

    #end of training
    #############################################################################################################################################################
    log.info('Sentiment')

    train_arrays_d2v = numpy.zeros((200, 100))
    train_labels_d2v = numpy.zeros(200)
    for i in range(100):
        prefix_train_pos = 'TRAIN_POS_' + str(i)
        prefix_train_neg = 'TRAIN_NEG_' + str(i)
        train_arrays_d2v[i] = model.docvecs[prefix_train_pos]
        train_arrays_d2v[100 + i] = model.docvecs[prefix_train_neg]
        train_labels_d2v[i] = 1
        train_labels_d2v[100 + i] = 0
    log.info(train_labels_d2v)

    test_arrays_d2v = numpy.zeros((200, 100))
    test_labels_d2v = numpy.zeros(200)
    for i in range(100):
        prefix_test_pos = 'TEST_POS_' + str(i)
        prefix_test_neg = 'TEST_NEG_' + str(i)
        test_arrays_d2v[i] = model.docvecs[prefix_test_pos]
        test_arrays_d2v[100 + i] = model.docvecs[prefix_test_neg]
        test_labels_d2v[i] = 1
        test_labels_d2v[100 + i] = 0
    log.info(test_labels_d2v)

    ################################################################################################################################################################
    #logistic regression classifier
    log.info('Fitting')
    classifier = LogisticRegression()
    classifier.fit(train_arrays_d2v, train_labels_d2v)
    #Result
    log.info(classifier.score(test_arrays_d2v, test_labels_d2v))
    ################################################################################################################################################################
    #CONFUSION MATRIX
    test_pred_labels_d2v = classifier.predict(test_arrays_d2v)
    test_pred_prob_d2v = classifier.predict_proba(test_arrays_d2v)[:, 1]
    ##CONFUSION MATRIX
    print("\nConfusion matrix:\n")
    confusion = metrics.confusion_matrix(test_labels_d2v, test_pred_labels_d2v)
    print(confusion)
    score[x]=classifier.score(test_arrays_d2v, test_labels_d2v)
    conf[x]=confusion

t=list(zip(score,conf))
z=list(zip(s,t))

d= {key: value for (key, value) in z}
df2.append(pd.DataFrame(data=d,index=['accuracy score','confusion matrix']))
###############################################################################################################################################
s=[0,1,2,5,20,100]
for x in range(len(s)):
    model = Doc2Vec(dm =0, dbow_words=1, min_count=1, window=10, size=100, sample=1e-4, negative=s[x], workers=8)
    model.build_vocab(corp)
    for epoch in range(5):
        log.info('Epoch %d' % epoch)
        model.train(sentences.sentences_perm(),
                    total_examples=model.corpus_count,
                    epochs=model.iter,)

    #end of training
    #############################################################################################################################################################
    log.info('Sentiment')

    train_arrays_d2v = numpy.zeros((200, 100))
    train_labels_d2v = numpy.zeros(200)
    for i in range(100):
        prefix_train_pos = 'TRAIN_POS_' + str(i)
        prefix_train_neg = 'TRAIN_NEG_' + str(i)
        train_arrays_d2v[i] = model.docvecs[prefix_train_pos]
        train_arrays_d2v[100 + i] = model.docvecs[prefix_train_neg]
        train_labels_d2v[i] = 1
        train_labels_d2v[100 + i] = 0
    log.info(train_labels_d2v)

    test_arrays_d2v = numpy.zeros((200, 100))
    test_labels_d2v = numpy.zeros(200)
    for i in range(100):
        prefix_test_pos = 'TEST_POS_' + str(i)
        prefix_test_neg = 'TEST_NEG_' + str(i)
        test_arrays_d2v[i] = model.docvecs[prefix_test_pos]
        test_arrays_d2v[100 + i] = model.docvecs[prefix_test_neg]
        test_labels_d2v[i] = 1
        test_labels_d2v[100 + i] = 0
    log.info(test_labels_d2v)

    ################################################################################################################################################################
    #logistic regression classifier
    log.info('Fitting')
    classifier = LogisticRegression()
    classifier.fit(train_arrays_d2v, train_labels_d2v)
    #Result
    log.info(classifier.score(test_arrays_d2v, test_labels_d2v))
    ################################################################################################################################################################
    #CONFUSION MATRIX
    test_pred_labels_d2v = classifier.predict(test_arrays_d2v)
    test_pred_prob_d2v = classifier.predict_proba(test_arrays_d2v)[:, 1]
    ##CONFUSION MATRIX
    print("\nConfusion matrix:\n")
    confusion = metrics.confusion_matrix(test_labels_d2v, test_pred_labels_d2v)
    print(confusion)
    score[x]=classifier.score(test_arrays_d2v, test_labels_d2v)
    conf[x]=confusion

t=list(zip(score,conf))
z=list(zip(s,t))

d= {key: value for (key, value) in z}
df2.append(pd.DataFrame(data=d,index=['accuracy score','confusion matrix']))
###################################################################################################################################################################
s=[1,5,20,100]
for x in range(len(s)):
    model = Doc2Vec(dm =0, dbow_words=1, min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=8)
    model.build_vocab(corp)
    for epoch in range(s[x]):
        log.info('Epoch %d' % epoch)
        model.train(sentences.sentences_perm(),
                    total_examples=model.corpus_count,
                    epochs=model.iter,)

    #end of training
    #############################################################################################################################################################
    log.info('Sentiment')

    train_arrays_d2v = numpy.zeros((200, 100))
    train_labels_d2v = numpy.zeros(200)
    for i in range(100):
        prefix_train_pos = 'TRAIN_POS_' + str(i)
        prefix_train_neg = 'TRAIN_NEG_' + str(i)
        train_arrays_d2v[i] = model.docvecs[prefix_train_pos]
        train_arrays_d2v[100 + i] = model.docvecs[prefix_train_neg]
        train_labels_d2v[i] = 1
        train_labels_d2v[100 + i] = 0
    log.info(train_labels_d2v)

    test_arrays_d2v = numpy.zeros((200, 100))
    test_labels_d2v = numpy.zeros(200)
    for i in range(100):
        prefix_test_pos = 'TEST_POS_' + str(i)
        prefix_test_neg = 'TEST_NEG_' + str(i)
        test_arrays_d2v[i] = model.docvecs[prefix_test_pos]
        test_arrays_d2v[100 + i] = model.docvecs[prefix_test_neg]
        test_labels_d2v[i] = 1
        test_labels_d2v[100 + i] = 0
    log.info(test_labels_d2v)

    ################################################################################################################################################################
    #logistic regression classifier
    log.info('Fitting')
    classifier = LogisticRegression()
    classifier.fit(train_arrays_d2v, train_labels_d2v)
    #Result
    log.info(classifier.score(test_arrays_d2v, test_labels_d2v))
    ################################################################################################################################################################
    #CONFUSION MATRIX
    test_pred_labels_d2v = classifier.predict(test_arrays_d2v)
    test_pred_prob_d2v = classifier.predict_proba(test_arrays_d2v)[:, 1]
    ##CONFUSION MATRIX
    print("\nConfusion matrix:\n")
    confusion = metrics.confusion_matrix(test_labels_d2v, test_pred_labels_d2v)
    print(confusion)
    score[x]=classifier.score(test_arrays_d2v, test_labels_d2v)
    conf[x]=confusion

t=list(zip(score,conf))
z=list(zip(s,t))

d= {key: value for (key, value) in z}
df2.append(pd.DataFrame(data=d,index=['accuracy score','confusion matrix']))
#################################################################################################################################################################
pvdbow_1=pd.concat(df2,keys=['dimention of vector','random sampling','negative sampling','epochs'],axis=1)
pvdbow_1.to_excel(writer,sheet_name='pvdbow_1',  merge_cells=True, startrow=0, startcol=0)
writer.save()
##################################################################################################################################################################
#########################################################
x=list(df2[0].columns)
y=list(df2[0].values[0])

plt.plot(x, y)
plt.title('accuracy vs dim')
plt.xlabel('dimention of vector')
plt.ylabel('accuracy score')
plt.ylim([0.0, 1.0])
plt.grid(True)
for i in range(len(x)):
    plt.annotate("("+str(x[i])+","+str(y[i])+")",xy=(x[i],y[i]))
####################################################

x=list(df2[1].columns)
y=list(df2[1].values[0])

plt.plot(x, y,'b-d')
plt.title('accuracy vs sub-sampling rate')
plt.xlabel('sub-sampling rate')
plt.ylabel('accuracy score')
plt.ylim([0.0, 1.0])
plt.grid(True)
for i in range(len(x)):
    plt.annotate("("+str(x[i])+","+str(y[i])+")",xy=(x[i],y[i]))
############################################
x=list(df2[2].columns)
y=list(df2[2].values[0])

plt.plot(x, y)
plt.title('accuracy vs negative sampling rate')
plt.xlabel('negative sampling rate')
plt.ylabel('accuracy score')
plt.ylim([0.0, 1.0])
plt.grid(True)
for i in range(len(x)):
    plt.annotate("("+str(x[i])+","+str(y[i])+")",xy=(x[i],y[i]))
########################################################################
x=list(df2[3].columns)
y=list(df2[3].values[0])

plt.plot(x, y)
plt.title('accuracy vs no. of epochs')
plt.xlabel('epochs')
plt.ylabel('accuracy score')
plt.ylim([0.0, 1.0])
plt.grid(True)
for i in range(len(x)):
    plt.annotate("("+str(x[i])+","+str(y[i])+")",xy=(x[i],y[i]))

