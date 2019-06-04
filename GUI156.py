import pandas as pd
print("Loading the first file")
df = pd.read_csv('text_emotion.csv', encoding='latin_1')
df.dropna()
df.head()
df_list = [df.columns.values.astype('U').tolist()] + df.values.tolist()
df_sentence = [df_list[x+1][1] for x in range(len(df_list)-1)]
emotion = [df_list[x+1][0] for x in range(len(df_list)-1)]
#print(emotion[0:10])
#print(type(emotion[0]))
#print(type('empty'))
#print(emotion[0] is emotion[526])
#print(len(emotion))
#print(df_sentence[0])
mylist = list(set(emotion))
print("Emotions in the first file: ")
print(mylist)
import numpy as np
print("Processing the first file")
valence_3 = np.zeros(len(emotion))
arousal_3 = np.zeros(len(emotion))
one_one = []
zero_zero = []
zero_one = []
one_zero = []
i = 0
j = 0
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
for i in range(j):
    valence3[i] = valence_3[i]
    arousal3[i] = arousal_3[i]
#print(len(valence3))
#print(len(arousal3))
df_sentence = one_one + zero_zero + zero_one + one_zero
print("Done processing the first file")
print("The total dataset of the first file is " + str(len(df_sentence)))
print("\nStart loading the second file")
emo = pd.read_csv('emo.csv', encoding='latin_1')
emo.dropna()
#fb = pd.read_csv('fbemo.csv')
#fb.dropna()
#fb_list = [fb.columns.values.astype('U').tolist()] + fb.values.tolist()
emo_list = [emo.columns.values.astype('U').tolist()] + emo.values.tolist()
valence_1 = np.zeros(len(emo_list)-1)
arousal_1 = np.zeros(len(emo_list)-1)
print("Processing the second file")
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
'''
valence_2 = np.zeros(len(fb_list)-1)
arousal_2 = np.zeros(len(fb_list)-1)
i = 1
while i < int(len(fb_list)):
    if (fb_list[i][1] + fb_list[i][2])/2 >= 5:
        valence_2[i-1] = 1 #for positive valence
    else:
        valence_2[i-1] = 0 #for negative valence
    if (fb_list[i][3] + fb_list[i][4])/2 >= 5:
        arousal_2[i-1] = 1 #for positive arousal
    else:
        arousal_2[i-1] = 0 #for negative arousal
    i+=1
'''
print("Done processing the second file")
#print(valence_1.shape)
#print(valence_2.shape)
emo_sentence = [emo_list[x+1][0] for x in range(len(emo_list)-1)]
#fb_sentence = [fb_list[x+1][0] for x in range(len(fb_list)-1)]
#sentence = emo_sentence + fb_sentence + df_sentence
sentence = emo_sentence + df_sentence
valence_np = np.hstack((valence_1, valence3)) #, valence_2
arousal_np = np.hstack((arousal_1, arousal3)) #, arousal_2
valence = valence_np.tolist()
arousal = arousal_np.tolist()
from sklearn.model_selection import train_test_split
#S_train = [sentence[i].value().astype(U) for i in range(len(sentence))]
S_train = sentence
v_train = valence
a_train = arousal
print("The total length of the dataset is " + str(len(S_train)))
#print(len(v_train))
#print(len(a_train))
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
middle_words = ['and','a','the','am','it','me','with','in','on','by','near','this','that','an','there','here','those','ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once',
                'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into',
                'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves',
                'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more',
                'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she',
                'all', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves',
                'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'myself', 'which', 'those', 'i',
                'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']
middle_words = set(dict.fromkeys([stemmer.stem(word) for word in middle_words]))
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
def tokenize(sentence):
    tokens = tokenizer.tokenize(sentence)
    for w in middle_words:
        while w in tokens:
            tokens.remove(w)
    toReturn = [stemmer.stem(item.lower()) for item in tokens]
    return toReturn
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
                'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren']
    middle_words = set(dict.fromkeys([stemmer.stem(word) for word in middle_words]))
    tokens = tokenizer.tokenize(sentence)
    for w in middle_words:
        while w in tokens:
            tokens.remove(w)
    toReturn = [stemmer.stem(item.lower()) for item in tokens]
    return toReturn
print("\n Preprocessing the data...")
from sklearn.feature_extraction.text import TfidfVectorizer
count_vect = TfidfVectorizer(ngram_range=(1,3), tokenizer=tokenize)
trainX = count_vect.fit_transform(S_train)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(v_train)
target_labels = le.classes_
trainy = le.transform(v_train)
print("\n Define tuning functions...")
def train_random(X,y):
    from sklearn.linear_model import RandomizedLogisticRegression
    randomized_logistic = RandomizedLogisticRegression()
    randomized_logistic.fit(X, y)
    print("Parameters: ", randomized_logistic.get_params)
    print("Score: ", str(randomized_logistic.score(X,y)))
    return randomized_logistic

def train_bagging(X,y):
    from sklearn.ensemble import BaggingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    bagging = BaggingClassifier(base_estimator = LogisticRegression(C=5, random_state=0, solver='lbfgs',class_weight = 'balanced', max_iter=10000))
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

def train_classifier(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    param_grid = {'C': [1, 5, 10, 25]}
    print("grid search start")
    grid = GridSearchCV(LogisticRegression(random_state=0, solver='lbfgs',class_weight = 'balanced', max_iter=10000), param_grid, cv=5)
    print("done grid search")
    grid.fit(X, y)
    print("Best cross-validation score: {:.2f}".format(grid.best_score_))
    print("Best parameters: ", grid.best_params_)
    print("Best estimator: ", grid.best_estimator_)
    cls = grid.best_estimator_
    cls.fit(X, y)
    return cls
print("Done preprocessing\n")
print("Start Training valence classifier")
cls_valence = train_classifier(trainX, trainy)
#svc_valence = train_svc(trainX, trainy)
#bagging_valence = train_bagging(trainX, trainy)
#cls_random = train_random_forest(trainX, trainy)
le_a = preprocessing.LabelEncoder()
le_a.fit(a_train)
target_labels_a = le_a.classes_
trainy = le_a.transform(a_train)
print("Done training valence\n")
print("Start Training arousal classifier")
le_a = preprocessing.LabelEncoder()
cls_arousal = train_classifier(trainX, trainy)
print("\nDone\n")
print("______________________________________")
test_list = ['The food is not good, but the music is nice and service is fine']
test = count_vect.transform(test_list)
lr_v = cls_valence.predict(test)
lr_a = cls_arousal.predict(test)
print(test_list)
print("Out prediction is: ")
if(lr_v == 1 and lr_a == 1):
    print("You are Happy")
elif(lr_v == 1 and lr_a == 0):
    print("You are just Chilling")
elif(lr_v == 0 and lr_a == 1):
    print(" You are really displeased")
elif(lr_v == 0 and lr_a == 0):
    print("You are bored or you are sad")

print("\n\n\n\nREADY TO ROLL!!!\n")
import tkinter as tk
from PIL import Image
#this is the application class for the GUI
class Application(tk.Frame):
    #TODO
    #here is the prediction input; we are going to have one input at a time

    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.count = 0

        #this is the title
        self.winfo_toplevel().title("CSE 156 Final Project\n")

        #this is the first button to train the two classifiers
        self.train = tk.Button(self, text = "Example", fg="red")
        self.train["command"] = self.example
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
        test_list[0] = self.label.get()
        print(test_list[0])

    def example(self):
        print()
        print("_____________________________________________")
        print("\n\nTHIS IS AN EXAMPLE\n")
        self.test = count_vect.transform(test_list)
        self.lr_v = cls_valence.predict(self.test)
        self.lr_a = cls_arousal.predict(self.test)
        print("Your input is")
        print(test_list[0])
        print("Our prediction is:")
        print("Should be: You are just Chilling")
        print(self.lr_v)
        print(self.lr_a)
        if(self.lr_v == 1 and self.lr_a == 1):
            pil_im = Image.open('/Users/apple/Desktop/CSE 156/final_cse156/happy.jpeg', 'r')
            print("\tYou are Happy")
        elif(self.lr_v == 1 and self.lr_a == 0):
            pil_im = Image.open('/Users/apple/Desktop/CSE 156/final_cse156/chilling.jpeg', 'r')
            print("\tYou are just Chilling")
        elif(self.lr_v == 0 and self.lr_a == 1):
            pil_im = Image.open('/Users/apple/Desktop/CSE 156/final_cse156/angry.jpeg', 'r')
            print("\tYou are really Displeased or Pissed")
        elif(self.lr_v == 0 and self.lr_a == 0):
            pil_im = Image.open('/Users/apple/Desktop/CSE 156/final_cse156/sad.jpeg', 'r')
            print("\tYou are Bored or you are Sad")
        pil_im.show()

    def predict(self):
        print("Predicting........")
        self.test = count_vect.transform(test_list)
        self.lr_v = cls_valence.predict(self.test)
        self.lr_a = cls_arousal.predict(self.test)
        print(self.lr_v)
        print(self.lr_a)
        print("Your input is")
        print("\t" + test_list[0])
        self.count += 1
        print("Our prediction is:")
        if(self.lr_v == 1 and self.lr_a == 1):
            pil_im = Image.open('/Users/apple/Desktop/CSE 156/final_cse156/happy.jpeg', 'r')
            print("\tYou are Happy")
        elif(self.lr_v == 1 and self.lr_a == 0):
            pil_im = Image.open('/Users/apple/Desktop/CSE 156/final_cse156/chilling.jpeg', 'r')
            print("\tYou are just Chilling")
        elif(self.lr_v == 0 and self.lr_a == 1):
            pil_im = Image.open('/Users/apple/Desktop/CSE 156/final_cse156/angry.jpeg', 'r')
            print("\tYou are really Displeased or Pissed")
        elif(self.lr_v == 0 and self.lr_a == 0):
            pil_im = Image.open('/Users/apple/Desktop/CSE 156/final_cse156/sad.jpeg', 'r')
            print("\tYou are Bored or you are Sad")
        pil_im.show()


root = tk.Tk()
root.geometry("400x300")
app = Application(master=root)
app.mainloop()

