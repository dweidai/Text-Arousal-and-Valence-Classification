import tkinter as tk
import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

'''
The goal of the project is to do a multi-class emotion classification
We will accomplish the goal by first classfy the input's valence then arousal
We will mark 0 for negative and 1 for postive.
For example, valence is 1 and arousal is 0 means
Positive valence and negative arousal --> chilling / relaxed
For more information,
you may visit https://www.researchgate.net/figure/The-2D-valence-arousal-emotion-space-Russell-1980-the-position-of-the-affective_fig1_254004106
'''

#this is the training for randomized logistic regression
#positivity: higher accuracy
#negativity: high computational power and warning from the compiler due to delay
def train_random(X,y):
    from sklearn.linear_model import RandomizedLogisticRegression
    randomized_logistic = RandomizedLogisticRegression()
    randomized_logistic.fit(X, y)
    print("Parameters: ", randomized_logistic.get_params)
    print("Score: ", str(randomized_logistic.score(X,y)))
    return randomized_logistic

#this is the training for bagging classification with logistic regression classfier
#positivity: high accuracy
#negativity: long time, really long time
def train_bagging(X,y):
    bagging = BaggingClassifier(base_estimator = LogisticRegression(C=5, random_state=0, solver='lbfgs',
            class_weight = 'balanced', max_iter=10000))
    f_list = [0.25,0.5,0.75]
    parameters_bagging = {'max_features': f_list}
    grid = GridSearchCV(bagging, parameters_bagging, cv=5)
    grid.fit(X, y)
    print("Best cross-validation score: {:.2f}".format(grid.best_score_))
    print("Best parameters: ", grid.best_params_)
    print("Best estimator: ", grid.best_estimator_)
    cls_bagging = grid.best_estimator_
    cls_bagging.fit(X, y)
    return cls_bagging

#this is used the tokenize function
#basically will remove all the middle words to manually remove some of the ambiguous words
def tokenize(sentence):
    tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
    stemmer = SnowballStemmer("english")

    middle_words = ['and','a','the','am','it','me','with','in','on','by','near','this','that','an','there','here','those',
                'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once','during', 'out', 'very',
                'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most',
                'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves',
                'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more',
                'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she',
                'all', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',
                'yourselves', 'then', 'that', 'because', 'what', 'over', 'so', 'can', 'did', 'now', 'under', 'he', 'you',
                'herself', 'has', 'just', 'where', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if',
                'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than', 's', 't', 'can', 'will',
                'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't",
                'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
                'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn',
                "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    middle_words = set(dict.fromkeys([stemmer.stem(word) for word in middle_words]))
    tokens = tokenizer.tokenize(sentence)
    for w in middle_words:
        while w in tokens:
            tokens.remove(w)
    toReturn = [stemmer.stem(item.lower()) for item in tokens]
    return toReturn

#this is the basic logistic regression
#same as the PA2
def train_classifier_valence(X, y):
    '''
    param_grid = {'C': [1, 5, 10, 25]}
    print("grid search start")
    grid = GridSearchCV(LogisticRegression(random_state=0, solver='lbfgs',class_weight = 'balanced', max_iter=10000),
                param_grid, cv=5)
    print("done grid search")
    grid.fit(X, y)
    print("Best cross-validation score: {:.2f}".format(grid.best_score_))
    print("Best parameters: ", grid.best_params_)
    print("Best estimator: ", grid.best_estimator_)
    cls = grid.best_estimator_
    '''
    #manually save the parameters
    cls = LogisticRegression(C=5, class_weight='balanced', dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=10000,
          multi_class='warn', n_jobs=None, penalty='l2', random_state=0,
          solver='lbfgs', tol=0.0001, verbose=0, warm_start=False)
    cls.fit(X, y)
    return cls

def train_classifier_arousal(X, y):
    cls = LogisticRegression(C=10, class_weight='balanced', dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=10000,
          multi_class='warn', n_jobs=None, penalty='l2', random_state=0,
          solver='lbfgs', tol=0.0001, verbose=0, warm_start=False)
    cls.fit(X,y)
    return cls

#this is the application class for the GUI
class Application(tk.Frame):
    #here is the prediction input; we are going to have one input at a time
    #here is the prediction input; we are going to have one input at a time

    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.count = 0
        self.test_list = ['The food is not good, but the music is nice and service is fine']
        #here we declare two classification functions
        self.cls_valence = LogisticRegression(C=5, class_weight='balanced', dual=False, fit_intercept=True,
                intercept_scaling=1, max_iter=10000, multi_class='warn', n_jobs=None, penalty='l2',
                random_state=0, solver='lbfgs', tol=0.0001, verbose=0, warm_start=False)

        self.cls_arousal = LogisticRegression(C=5, class_weight='balanced', dual=False, fit_intercept=True,
                intercept_scaling=1, max_iter=10000,multi_class='warn', n_jobs=None, penalty='l2',
                random_state=0,solver='lbfgs', tol=0.0001, verbose=0, warm_start=False)

        #this is the title
        self.winfo_toplevel().title("CSE 156 Final Project\n")

        #this is the first button to train the two classifiers
        self.train = tk.Button(self, text = "Click to Train", fg="red")
        self.train["command"] = self.train_classifier
        self.train.pack(side="top")

        #here is the instruction
        self.text = tk.Label(self, text=" Please input the sentence you wnat to predict: ")
        self.text.pack()
        #entry here
        self.label = tk.Entry(self, bd=5)
        self.label.pack(side="top", fill="x")

        #print(label.get())
        #print(type(label.get()))
        self.ok = tk.Button(self, text = "Confirm Your Phrase", fg="red")
        self.ok["command"] = self.confirm
        self.ok.pack(side = "top")

        self.run = tk.Button(self, fg = "red")
        self.run["text"] = "Click to Predict\n"
        self.run["command"] = self.predict
        self.run.pack(side="top")


        self.quit = tk.Button(self, text="QUIT", fg="red",
                              command=self.master.destroy)
        self.quit.pack(side="bottom")

    def confirm(self):
        self.test_list[0] = self.label.get()
        print(self.test_list)

    def train_classifier(self):
        print("Training starts\n")

        df = pd.read_csv('/Users/apple/Desktop/CSE 156/final_cse156/text_emotion.csv', encoding='latin_1')
        df.dropna()
        df_list = [df.columns.values.astype('U').tolist()] + df.values.tolist()
        df_sentence = [df_list[x+1][1] for x in range(len(df_list)-1)]
        emotion = [df_list[x+1][0] for x in range(len(df_list)-1)]
        valence_3 = np.zeros(len(emotion))
        arousal_3 = np.zeros(len(emotion))
        one_one = []
        zero_zero = []
        zero_one = []
        one_zero = []
        i = 0
        j = 0
        print("Processing first file")
        while i < int(len(emotion)):
            if(emotion[i] == 'fun' or emotion[i] == 'happiness' or emotion[i] == 'enthusiasm' or emotion[i] == 'love'):
                valence_3[j] = 1
                arousal_3[j] = 1
                one_one.append(df_sentence[i])
                j+=1
            i += 1
        i = 0
        while i < int(len(emotion)):
            if(emotion[i] == 'sad' or emotion[i] == 'boredom' or emotion[i] == 'worry'):
                valence_3[j] = 0
                arousal_3[j] = 0
                zero_zero.append(df_sentence[i])
                j+=1
            i += 1
        i = 0
        while i < int(len(emotion)):
            if(emotion[i] == 'anger' or emotion[i] == 'hate'):
                valence_3[j] = 0
                arousal_3[j] = 1
                zero_one.append(df_sentence[i])
                j+=1
            i += 1
        i = 0
        while i < int(len(emotion)):
            if(emotion[i] == 'neural' or emotion[i] == 'relief'):
                valence_3[j] = 1
                arousal_3[j] = 0
                one_zero.append(df_sentence[i])
                j+=1
            i+=1
        valence3 = np.zeros(j)
        arousal3 = np.zeros(j)
        df_sentence = one_one + zero_zero + zero_one + one_zero
        print("Done processing first file\n")
        print("Processing second file")
        emo = pd.read_csv('emo.csv', encoding='latin_1')
        emo.dropna()
        emo_list = [emo.columns.values.astype('U').tolist()] + emo.values.tolist()
        valence_1 = np.zeros(len(emo_list)-1)
        arousal_1 = np.zeros(len(emo_list)-1)
        i = 1
        while i < int(len(emo_list)):
            if emo_list[i][1] >= 3:
                valence_1[i-1] = 1 #for positive valence
            else:
                valence_1[i-1] = 0 #for negative valence
            if emo_list[i][2] >= 3:
                arousal_1[i-1] = 1 #for positive arousal
            else:
                arousal_1[i-1] = 0 #for negative arousal
            i+=1
        emo_sentence = [emo_list[x+1][0] for x in range(len(emo_list)-1)]
        sentence = emo_sentence + df_sentence
        valence_np = np.hstack((valence_1, valence3)) #, valence_2
        arousal_np = np.hstack((arousal_1, arousal3)) #, arousal_2
        print("Done processing second file\n")

        valence = valence_np.tolist()
        arousal = arousal_np.tolist()
        self.S_train = sentence
        v_train = valence
        a_train = arousal

        print("Start Preprocessing...")
        count_vect = TfidfVectorizer(ngram_range=(1,3), tokenizer=tokenize)
        trainX = count_vect.fit_transform(self.S_train)
        le = preprocessing.LabelEncoder()
        le.fit(v_train)
        target_labels = le.classes_
        trainy = le.transform(v_train)
        print("Done Preprocessing")
        print("Start Training for valence classification")
        self.cls_valence = train_classifier_valence(trainX, trainy)
        le_a = preprocessing.LabelEncoder()
        le_a.fit(a_train)
        target_labels_a = le_a.classes_
        trainy = le_a.transform(a_train)
        le_a = preprocessing.LabelEncoder()
        print("Done valence classification")
        print("Start Training for arousal classification")
        self.cls_arousal = train_classifier_arousal(trainX, trainy)
        print("Done Training")
        print()
        print("_____________________________________________")
        print("\n\nTHIS IS AN EXAMPLE\n")
        test = count_vect.transform(self.test_list)
        lr_v = self.cls_valence.predict(test)
        lr_a = self.cls_arousal.predict(test)
        print("Your input is")
        print(self.test_list[0])
        print("Our prediction is:")
        if(lr_v == 1 and lr_a == 1):
            print("\tYou are Happy")
        elif(lr_v == 1 and lr_a == 0):
            print("\tYou are just Chilling")
        elif(lr_v == 0 and lr_a == 1):
            print("\tYou are really Displeased or Pissed")
        elif(lr_v == 0 and lr_a == 0):
            print("\tYou are Bored or you are Sad")


    def predict(self):
        count_vect = TfidfVectorizer(ngram_range=(1,3), tokenizer=tokenize)
        print("fitting to model")
        trainX = count_vect.fit_transform(self.S_train)
        print("Predicting........")
        test = count_vect.transform(self.test_list)
        lr_v = self.cls_valence.predict(test)
        lr_a = self.cls_arousal.predict(test)
        print("Your input is")
        print("\t" + self.test_list[0])
        self.count += 1
        print("Our prediction is:")
        if(lr_v == 1 and lr_a == 1):
            print("\tYou are Happy")
        elif(lr_v == 1 and lr_a == 0):
            print("\tYou are just Chilling")
        elif(lr_v == 0 and lr_a == 1):
            print("\tYou are really Displeased or Pissed")
        elif(lr_v == 0 and lr_a == 0):
            print("\tYou are Bored or you are Sad")

root = tk.Tk()
root.geometry("400x300")
app = Application(master=root)
app.mainloop()
