import re
import math
import time

training_time = time.time()

f =  open('/scratch/ds222-2017/assignment-1/DBPedia.full/full_train.txt', 'r') 
cls = set()
text_samples = []
label_samples = []
for article in f:
    try :        
        labels, line = article.split('\t')
    except :
        continue
    labels = labels.split(',')
    line = [ i.replace(" ","").lower() for i in re.sub(r'[^a-zA-Z]', " ",re.sub(r'<.*>',"", line) ).split()[:-1] if len(i) > 3]
    line = ' '.join(line)
    text_samples.append(line)
    labels = [label.replace(" ","") for label in labels]
    label_samples.append(labels)
    for label in labels:
        cls.add(label)
f.close()


k = 0
A = {}
W_class ={}
for c in list(cls):
    W_class[c] = {}
    A[c] ={}

##DF##
feature = []

df = {}
for line in text_samples:
    tf = {}
    for word in line.split():
        try :
            tf[word] = tf[word] + 1
        except :
            tf[word] = 1
            
    for key in tf.keys() :
        try :
            df[key] = df[key] + 1
        except :
            df[key] = 1
            
    feature.append(tf.items())

Nfeatures = 2000 

import operator

sorted_x = sorted(df.items(), key= operator.itemgetter(1))
df = dict(sorted_x[-Nfeatures:])

N_docs = len(feature)


tfidf = []

tfidf = []

for i, sample in enumerate(feature):
    s = []
    for word in sample :
        try :
            s.append( (word[0], math.log(1+word[1]) * math.log(N_docs * 1./df[word[0]]) ) )
        except :
            continue
    sample = s
    scale  = sum([ word[1]**2 for word in sample]) **0.5
    sample = [ (word[0], word[1]/scale) for word in sample]
    tfidf.append( sample )
    
feature = tfidf

for cls in W_class.keys():
    for word in df.keys():
        W_class[cls][word] = 0
        A[cls][word] = 0

def prob_class(cls, x):
    score = 0
    for word in x:
        score = score + W_class[cls][word[0]] * word[1]
    p = 1./(1+math.exp(-1*score))
    return p


def loss_function(features, label_samples):
    loss = 0
    labels = W_class.keys()
    for i in range(len(features)):
        x = features[i]
        p_class = []
        z = 0
        for label in labels:
            p = prob_class(label, x)
            p_class.append((label, p))
            z = z + p
        for label, p in p_class :
            if label in label_samples[i]:
                loss = loss + -1* math.log(p/z)
            loss = loss / len(label_samples[i])
            
    return loss/ len(features)


lbda = 0.1
cls = W_class.keys()
mu = 0.01
temp_loss = 0

print "loss per epoch" 

loss_per_epoch = []
for epoch in range(30):
    k = k + 1
    time_loss = time.time()
    
    loss_per_epoch.append(loss_function(feature, label_samples))

    print epoch, loss_per_epoch[-1]
    
    temp_loss = temp_loss + time.time()-time_loss
    for i in range( len(feature)):
        x = feature[i]
        for label in cls:
            y = 0
            if label  in  label_samples[i]:
                y = 1
            p = prob_class(label, x)
            for word in x:
                W_class[label][word[0]] = W_class[label][word[0]] * ( (1- 2* mu * lbda) ** (k-A[label][word[0]]))
                W_class[label][word[0]] = W_class[label][word[0]] + lbda * (y-p) * word[1]
                A[label][word[0]] = k



def testing_accuracy(feature_test, label_test):
    labels = W_class.keys()
    acc = 0
    for i in range(len(feature_test)):
        x = feature_test[i]
        p_max = 0
        
        for label in labels:
            p = prob_class(label, x)
            if p > p_max :
                label_predict = label
                p_max = p
        if label_predict in label_test[i]:
            acc = acc + 1
    return acc *1./len(feature_test)

print "Training time", time.time() - training_time  - temp_loss

print "Training Accuracy"

print testing_accuracy(feature, label_samples)

testing_time = time.time()
f =  open('/scratch/ds222-2017/assignment-1/DBPedia.full/full_test.txt', 'r') 

text_test = []
label_test = []
for article in f:
    try :        
        labels, line = article.split('\t')
    except :
        continue
    labels = labels.split(',')
    line = [ i.replace(" ","").lower() for i in re.sub(r'[^a-zA-Z]', " ",re.sub(r'<.*>',"", line) ).split()[:-1] if len(i) > 3]
    line = ' '.join(line)
    text_test.append(line)
    labels = [label.replace(" ","") for label in labels]
    label_test.append(labels)
    
f.close()

feature = []

for line in text_test:
    tf = {}
    for word in line.split():
        try :
            tf[word] = tf[word] + 1
        except :
            tf[word] = 1
    feature.append(tf.items())


tfidf = []

for i, sample in enumerate(feature):
    s = []
    for word in sample :
        try :
            s.append( (word[0], math.log(1+word[1]) * math.log(N_docs * 1./df[word[0]]) ) )
        except :
            continue
    sample = s
    scale  = sum([ word[1]**2 for word in sample]) **0.5
    sample = [ (word[0], word[1]/scale) for word in sample]
    tfidf.append( sample )

feature = tfidf



print "Testing Accuracy"
print testing_accuracy(feature, label_test)

print "Testing time", time.time()-testing_time

    





            





