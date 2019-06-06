#!/usr/bin/env python
# coding: utf-8
import numpy as np

# In[180]:


from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
middle_words = ['and','a','the','am','it','me','with','in','on','by','near','this','that','an','there','here','those']
middle_words = set(dict.fromkeys([stemmer.stem(word) for word in middle_words]))


# In[181]:


def tokenize(sentence):
    tokens = tokenizer.tokenize(sentence)
    pick_word = tokens
    for w in middle_words:
        while w in pick_word:
            pick_word.remove(w)
    return tokens
def read_files(tarfname):
    """Read the training and development data from the sentiment tar file.
    The returned object contains various fields that store sentiment data, such as:

    train_data,dev_data: array of documents (array of words)
    train_fnames,dev_fnames: list of filenames of the doccuments (same length as data)
    train_labels,dev_labels: the true string label for each document (same length as data)

    The data is also preprocessed for use with scikit-learn, as:

    count_vec: CountVectorizer used to process the data (for reapplication on new data)
    trainX,devX: array of vectors representing Bags of Words, i.e. documents processed through the vectorizer
    le: LabelEncoder, i.e. a mapper from string labels to ints (stored for reapplication)
    target_labels: List of labels (same order as used in le)
    trainy,devy: array of int labels, one for each document
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    trainname = "train.tsv"
    devname = "dev.tsv"
    for member in tar.getmembers():
        if 'train.tsv' in member.name:
            trainname = member.name
        elif 'dev.tsv' in member.name:
            devname = member.name
            
            
    class Data: pass
    sentiment = Data()
    print("-- train data")
    sentiment.train_data, sentiment.train_labels = read_tsv(tar,trainname)
    print(len(sentiment.train_data))

    print("-- dev data")
    sentiment.dev_data, sentiment.dev_labels = read_tsv(tar, devname)
    print(len(sentiment.dev_data))
    print("-- transforming data and labels")
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    sentiment.count_vect = TfidfVectorizer(ngram_range=(1,3), tokenizer=tokenize) #CountVectorizer()
    sentiment.trainX = sentiment.count_vect.fit_transform(sentiment.train_data)
    sentiment.devX = sentiment.count_vect.transform(sentiment.dev_data)
    from sklearn import preprocessing
    sentiment.le = preprocessing.LabelEncoder()
    sentiment.le.fit(sentiment.train_labels)
    sentiment.target_labels = sentiment.le.classes_
    sentiment.trainy = sentiment.le.transform(sentiment.train_labels)
    sentiment.devy = sentiment.le.transform(sentiment.dev_labels)
    tar.close()
    return sentiment
def read_unlabeled(tarfname, sentiment):
    """Reads the unlabeled data.

    The returned object contains three fields that represent the unlabeled data.

    data: documents, represented as sequence of words
    fnames: list of filenames, one for each document
    X: bag of word vector for each document, using the sentiment.vectorizer
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    class Data: pass
    unlabeled = Data()
    unlabeled.data = []
    
    unlabeledname = "unlabeled.tsv"
    for member in tar.getmembers():
        if 'unlabeled.tsv' in member.name:
            unlabeledname = member.name
            
    print(unlabeledname)
    tf = tar.extractfile(unlabeledname)
    for line in tf:
        line = line.decode("utf-8")
        text = line.strip()
        unlabeled.data.append(text)
        
            
    unlabeled.X = sentiment.count_vect.transform(unlabeled.data)
    print(unlabeled.X.shape)
    tar.close()
    return unlabeled
def read_tsv(tar, fname):
    member = tar.getmember(fname)
    print(member.name)
    tf = tar.extractfile(member)
    data = []
    labels = []
    for line in tf:
        line = line.decode("utf-8")
        (label,text) = line.strip().split("\t")
        labels.append(label)
        data.append(text)
    return data, labels
def write_pred_kaggle_file(unlabeled, cls, outfname, sentiment):
    """Writes the predictions in Kaggle format.

    Given the unlabeled object, classifier, outputfilename, and the sentiment object,
    this function write sthe predictions of the classifier on the unlabeled data and
    writes it to the outputfilename. The sentiment object is required to ensure
    consistent label names.
    """
    yp = cls.predict(unlabeled.X)
    labels = sentiment.le.inverse_transform(yp)
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    for i in range(len(unlabeled.data)):
        f.write(str(i+1))
        f.write(",")
        f.write(labels[i])
        f.write("\n")
    f.close()
def write_gold_kaggle_file(tsvfile, outfname):
    """Writes the output Kaggle file of the truth.

    You will not be able to run this code, since the tsvfile is not
    accessible to you (it is the test labels).
    """
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    i = 0
    with open(tsvfile, 'r') as tf:
        for line in tf:
            (label,review) = line.strip().split("\t")
            i += 1
            f.write(str(i))
            f.write(",")
            f.write(label)
            f.write("\n")
    f.close()
def write_basic_kaggle_file(tsvfile, outfname):
    """Writes the output Kaggle file of the naive baseline.

    This baseline predicts POSITIVE for all the instances.
    """
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    i = 0
    with open(tsvfile, 'r') as tf:
        for line in tf:
            (label,review) = line.strip().split("\t")
            i += 1
            f.write(str(i))
            f.write(",")
            f.write("POSITIVE")
            f.write("\n")
    f.close()


# In[188]:


def train_classifier(X, y):
    """Train a classifier using the given training data.

    Trains logistic regression on the input data with default parameters.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    param_grid = {'C': [0.01, 0.05, 0.1, 0.15, 0.5, 1, 5, 10, 100, 200, 300, 400, 500]}
    grid = GridSearchCV(LogisticRegression(random_state=0, solver='lbfgs',class_weight = 'balanced', max_iter=10000), param_grid, cv=5)
    grid.fit(X, y)
    print("Best cross-validation score: {:.2f}".format(grid.best_score_))
    print("Best parameters: ", grid.best_params_)
    print("Best estimator: ", grid.best_estimator_)
    cls = grid.best_estimator_
    #cls = LogisticRegression(C=0.15, class_weight='balanced', dual=False,
    #      fit_intercept=True, intercept_scaling=1, max_iter=10000,
    #      multi_class='warn', n_jobs=None, penalty='l2', random_state=0,
    #      solver='lbfgs', tol=0.0001, verbose=0, warm_start=False)
    '''cls = LogisticRegression(C=5, class_weight='balanced', dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=10000,
          multi_class='warn', n_jobs=None, penalty='l2', random_state=0,
          solver='lbfgs', tol=0.0001, verbose=0, warm_start=False)'''
    #cls = LogisticRegression(random_state=0, solver='lbfgs', max_iter=10000)
    cls.fit(X, y)
    return cls


# In[184]:


def evaluate(X, yt, cls, name='data'):
    """Evaluated a classifier on the given labeled data using accuracy."""
    from sklearn import metrics
    yp = cls.predict(X)
    acc = metrics.accuracy_score(yt, yp)
    print("  Accuracy on %s  is: %s" % (name, acc))
    return acc


def check_result():
    from sklearn.pipeline import make_pipeline
    yp = cls.predict(sentiment.devX)
    for k in range(len(sentiment.dev_data)):
        if k < 100 and yp[k] != sentiment.devy[k]:
            print("\n", k)
            c = make_pipeline(sentiment.count_vect, cls)
            print(sentiment.dev_data[k])
            print(c.predict_proba([sentiment.dev_data[k]]))

            from lime.lime_text import LimeTextExplainer
            target = ['NEGATIVE', 'POSITIVE']
            explainer = LimeTextExplainer(class_names=target)
            exp = explainer.explain_instance(sentiment.dev_data[k], c.predict_proba, num_features=6)
            print('Document id: %d' % k)
            yp = cls.predict(sentiment.devX)
            print("real answer is: ", sentiment.devy[k])
            print("prediction is: ", yp[k])
            print('Probability(NEGATIVE) =', c.predict_proba([sentiment.dev_data[k]])[0, 0])
            print('True class: %s' % sentiment.dev_labels[k])
            msf = exp.as_list()
            print(msf)


def check_result2(sentence):
    predict = [sentence]
    transfered = sentiment.count_vect.transform(predict)
    from sklearn.pipeline import make_pipeline
    yp = cls.predict(transfered)
    for k in range(len(predict)):

        c = make_pipeline(sentiment.count_vect, cls)
        print("\n")
        print(predict[0])
        print(c.predict_proba([predict[0]]))

        from lime.lime_text import LimeTextExplainer
        target = ['NEGATIVE', 'POSITIVE']
        explainer = LimeTextExplainer(class_names=target)
        exp = explainer.explain_instance(predict[0], c.predict_proba, num_features=6)
        yp = cls.predict(transfered)
        print("prediction is: ", yp[0])
        print('Probability(NEGATIVE) =', c.predict_proba([predict[0]])[0, 0])
        msf = exp.as_list()
        print(msf)



# In[164]:


i = 1
max_acc = 0
max_i = 1
while(i<2):
    print(i)
    print("Reading data")
    tarfname = "data/sentiment.tar.gz"
    sentiment = read_files(tarfname)
    cls = train_classifier(sentiment.trainX, sentiment.trainy)
    coefficients=cls.coef_[0]
    length = len(coefficients)
    k = (int)(length/i)
    middle =np.argsort(coefficients)[k:-k]
    middle_words = []
    print("Colletcting Middle Ambiguous Words")
    #for j in middle:
     #   middle_words.append(sentiment.count_vect.get_feature_names()[j])
    #middle_words = set(dict.fromkeys([stemmer.stem(word) for word in middle_words]))
    print("Reading data")
    tarfname = "data/sentiment.tar.gz"
    sentiment = read_files(tarfname)
    cls = train_classifier(sentiment.trainX, sentiment.trainy)
    print("\nEvaluating")
    evaluate(sentiment.trainX, sentiment.trainy, cls, 'train')
    acc = evaluate(sentiment.devX, sentiment.devy, cls, 'dev')
    check_result()
    while True:
        val = input("Enter your sentence: ")
        check_result2(val)

#     if( acc > max_acc):
#         max_acc = acc
#         max_i = i
#     i = i+1
# print( max_i)
# print( max_acc)


# In[189]:


# print("Reading data")
# tarfname = "data/sentiment.tar.gz"
# sentiment = read_files(tarfname)
# cls = train_classifier(sentiment.trainX, sentiment.trainy)
# coefficients=cls.coef_[0]
# length = len(coefficients)
# k = (int)(length/max_i)
# middle =np.argsort(coefficients)[k:-k]
# middle_words = []
# print("Colletcting Middle Ambiguous Words")
# #for j in middle:
# #    middle_words.append(sentiment.count_vect.get_feature_names()[j])
# #middle_words = set(dict.fromkeys([stemmer.stem(word) for word in middle_words]))
# print("Reading data")
# tarfname = "data/sentiment.tar.gz"
# sentiment = read_files(tarfname)
# cls = train_classifier(sentiment.trainX, sentiment.trainy)
# print("\nEvaluating")
# evaluate(sentiment.trainX, sentiment.trainy, cls, 'train')
# acc = evaluate(sentiment.devX, sentiment.devy, cls, 'dev')
# check_result()

# In[191]:


# print("\nReading unlabeled data")
# unlabeled = read_unlabeled(tarfname, sentiment)
# print("Writing predictions to a file")
# write_pred_kaggle_file(unlabeled, cls, "data/sentiment-pred.csv", sentiment)
# print("done")

