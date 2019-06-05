""" Kivy GUI """
from kivy.app import App
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.textinput import TextInput
from kivy.logger import Logger
""" Computational GUI """
import pandas as pd
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from PIL import Image

""" Actual Computation Starts Here """

print("Loading the first file")
df = pd.read_csv('text_emotion.csv', encoding='latin_1')
df.dropna()
df.head()
df_list = [df.columns.values.astype('U').tolist()] + df.values.tolist()
df_sentence = [df_list[x + 1][1] for x in range(len(df_list) - 1)]
emotion = [df_list[x + 1][0] for x in range(len(df_list) - 1)]
# print(emotion[0:10])
# print(type(emotion[0]))
# print(type('empty'))
# print(emotion[0] is emotion[526])
# print(len(emotion))
# print(df_sentence[0])
mylist = list(set(emotion))
print("Emotions in the first file: ")
print(mylist)

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
	if (emotion[i] == 'fun' or emotion[i] == 'happiness' or emotion[i] == 'enthusiasm' or emotion[i] == 'love'):
		valence_3[j] = 1
		arousal_3[j] = 1
		one_one.append(df_sentence[i])
		j += 1
	i += 1
i = 0
while i < int(len(emotion)):
	if (emotion[i] == 'sad' or emotion[i] == 'boredom' or emotion[i] == 'worry'):
		valence_3[j] = 0
		arousal_3[j] = 0
		zero_zero.append(df_sentence[i])
		j += 1
	i += 1
i = 0
while i < int(len(emotion)):
	if (emotion[i] == 'anger' or emotion[i] == 'hate'):
		valence_3[j] = 0
		arousal_3[j] = 1
		zero_one.append(df_sentence[i])
		j += 1
	i += 1
i = 0
while i < int(len(emotion)):
	if (emotion[i] == 'neural' or emotion[i] == 'relief'):
		valence_3[j] = 1
		arousal_3[j] = 0
		one_zero.append(df_sentence[i])
		j += 1
	i += 1
valence3 = np.zeros(j)
arousal3 = np.zeros(j)
for i in range(j):
	valence3[i] = valence_3[i]
	arousal3[i] = arousal_3[i]
# print(len(valence3))
# print(len(arousal3))
df_sentence = one_one + zero_zero + zero_one + one_zero
print("Done processing the first file")
print("The total dataset of the first file is " + str(len(df_sentence)))
print("\nStart loading the second file")
emo = pd.read_csv('emo.csv', encoding='latin_1')
emo.dropna()
# fb = pd.read_csv('fbemo.csv')
# fb.dropna()
# fb_list = [fb.columns.values.astype('U').tolist()] + fb.values.tolist()
emo_list = [emo.columns.values.astype('U').tolist()] + emo.values.tolist()
valence_1 = np.zeros(len(emo_list) - 1)
arousal_1 = np.zeros(len(emo_list) - 1)
print("Processing the second file")
i = 1
while i < int(len(emo_list)):
	if emo_list[i][1] >= 3:
		valence_1[i - 1] = 1  # for positive valence
	else:
		valence_1[i - 1] = 0  # for negative valence
	if emo_list[i][2] >= 3:
		arousal_1[i - 1] = 1  # for positive arousal
	else:
		arousal_1[i - 1] = 0  # for negative arousal
	i += 1
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
# print(valence_1.shape)
# print(valence_2.shape)
emo_sentence = [emo_list[x + 1][0] for x in range(len(emo_list) - 1)]
# fb_sentence = [fb_list[x+1][0] for x in range(len(fb_list)-1)]
# sentence = emo_sentence + fb_sentence + df_sentence
sentence = emo_sentence + df_sentence
valence_np = np.hstack((valence_1, valence3))  # , valence_2
arousal_np = np.hstack((arousal_1, arousal3))  # , arousal_2
valence = valence_np.tolist()
arousal = arousal_np.tolist()
# S_train = [sentence[i].value().astype(U) for i in range(len(sentence))]
S_train = sentence
v_train = valence
a_train = arousal
print("The total length of the dataset is " + str(len(S_train)))
# print(len(v_train))
# print(len(a_train))

stemmer = SnowballStemmer("english")
middle_words = ['and', 'a', 'the', 'am', 'it', 'me', 'with', 'in', 'on', 'by', 'near', 'this', 'that', 'an', 'there',
                'here', 'those', 'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once',
                'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its',
                'yours', 'such', 'into',
                'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each',
                'the', 'themselves',
                'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her',
                'more',
                'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours',
                'had', 'she',
                'all', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',
                'yourselves',
                'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'now', 'under', 'he', 'you',
                'herself', 'has', 'just', 'where', 'myself', 'which', 'those', 'i',
                'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how',
                'further', 'was', 'here', 'than']
middle_words = set(dict.fromkeys([stemmer.stem(word) for word in middle_words]))

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
	middle_words = ['and', 'a', 'the', 'am', 'it', 'me', 'with', 'in', 'on', 'by', 'near', 'this', 'that', 'an',
	                'there', 'here', 'those',
	                'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during',
	                'out', 'very',
	                'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into',
	                'of', 'most',
	                'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the',
	                'themselves',
	                'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her',
	                'more',
	                'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours',
	                'had', 'she',
	                'all', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on',
	                'does',
	                'yourselves', 'then', 'that', 'because', 'what', 'over', 'so', 'can', 'did', 'now', 'under', 'he',
	                'you',
	                'herself', 'has', 'just', 'where', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't',
	                'being', 'if',
	                'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than', 's',
	                't', 'can', 'will',
	                'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain',
	                'aren']
	middle_words = set(dict.fromkeys([stemmer.stem(word) for word in middle_words]))
	tokens = tokenizer.tokenize(sentence)
	for w in middle_words:
		while w in tokens:
			tokens.remove(w)
	toReturn = [stemmer.stem(item.lower()) for item in tokens]
	return toReturn


print("\n Preprocessing the data...")

count_vect = TfidfVectorizer(ngram_range=(1, 3), tokenizer=tokenize)
trainX = count_vect.fit_transform(S_train)

le = preprocessing.LabelEncoder()
le.fit(v_train)
target_labels = le.classes_
trainy = le.transform(v_train)
print("\n Define tuning functions...")


def train_random(X, y):
	from sklearn.linear_model import RandomizedLogisticRegression
	randomized_logistic = RandomizedLogisticRegression()
	randomized_logistic.fit(X, y)
	print("Parameters: ", randomized_logistic.get_params)
	print("Score: ", str(randomized_logistic.score(X, y)))
	return randomized_logistic


def train_bagging(X, y):
	from sklearn.ensemble import BaggingClassifier
	from sklearn.linear_model import LogisticRegression
	from sklearn.model_selection import GridSearchCV
	bagging = BaggingClassifier(
		base_estimator=LogisticRegression(C=5, random_state=0, solver='lbfgs', class_weight='balanced', max_iter=10000))
	f_list = [0.25, 0.5, 0.75]
	parameters_bagging = {'max_features': f_list}
	grid = GridSearchCV(bagging, parameters_bagging, cv=5)
	grid.fit(X, y)
	print("Best cross-validation score: {:.2f}".format(grid.best_score_))
	print("Best parameters: ", grid.best_params_)
	print("Best estimator: ", grid.best_estimator_)
	cls_bagging = grid.best_estimator_
	cls_bagging.fit(X, y)
	return cls_bagging


def train_classifier_valence(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    '''param_grid = {'C': [1, 5, 10, 25]}
        print("grid search start")
        grid = GridSearchCV(LogisticRegression(random_state=0, solver='lbfgs', class_weight='balanced', max_iter=10000),
        param_grid, cv=5)
        print("done grid search")
        grid.fit(X, y)
        print("Best cross-validation score: {:.2f}".format(grid.best_score_))
        print("Best parameters: ", grid.best_params_)
        print("Best estimator: ", grid.best_estimator_)'''
    cls = LogisticRegression(C=5, class_weight='balanced', dual=False,
                             fit_intercept=True, intercept_scaling=1, max_iter=10000,
                             multi_class='warn', n_jobs=None, penalty='l2', random_state=0,
                             solver='lbfgs', tol=0.0001, verbose=0, warm_start=False)#grid.best_estimator_
    cls.fit(X, y)
    return cls

def train_classifier_arousal(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    '''param_grid = {'C': [1, 5, 10, 25]}
        print("grid search start")
        grid = GridSearchCV(LogisticRegression(random_state=0, solver='lbfgs', class_weight='balanced', max_iter=10000),
        param_grid, cv=5)
        print("done grid search")
        grid.fit(X, y)
        print("Best cross-validation score: {:.2f}".format(grid.best_score_))
        print("Best parameters: ", grid.best_params_)
        print("Best estimator: ", grid.best_estimator_)'''
    cls = LogisticRegression(C=10, class_weight='balanced', dual=False,
                             fit_intercept=True, intercept_scaling=1, max_iter=10000,
                             multi_class='warn', n_jobs=None, penalty='l2', random_state=0,
                             solver='lbfgs', tol=0.0001, verbose=0, warm_start=False)#grid.best_estimator_
    cls.fit(X, y)
    return cls



print("Done preprocessing\n")
print("Start Training valence classifier")
cls_valence = train_classifier_arousal(trainX, trainy)
# svc_valence = train_svc(trainX, trainy)
# bagging_valence = train_bagging(trainX, trainy)
# cls_random = train_random_forest(trainX, trainy)
le_a = preprocessing.LabelEncoder()
le_a.fit(a_train)
target_labels_a = le_a.classes_
trainy = le_a.transform(a_train)
print("Done training valence\n")
print("Start Training arousal classifier")
le_a = preprocessing.LabelEncoder()
cls_arousal = train_classifier_arousal(trainX, trainy)
print("\nDone\n")
print("______________________________________")
test_list = ['The food is not good, but the music is nice and service is fine']
test = count_vect.transform(test_list)
lr_v = cls_valence.predict(test)
lr_a = cls_arousal.predict(test)
print(test_list)
print("Out prediction is: ")
if (lr_v == 1 and lr_a == 1):
	print("You are Happy")
elif (lr_v == 1 and lr_a == 0):
	print("You are just Chilling")
elif (lr_v == 0 and lr_a == 1):
	print(" You are really displeased")
elif (lr_v == 0 and lr_a == 0):
	print("You are bored or you are sad")
print("\n\n\n\nREADY TO ROLL!!!\n")


class Application(App):
	"""
	GUI Class, using Kivy
	"""

	def build(self):
		"""
		Build up Application Graphical Interface
		:return: GUI window
		"""
		Window.size = (1000, 300)
		self.grid = FloatLayout()
		self.box_row1 = BoxLayout(orientation='horizontal', pos_hint={"y": 0.6},
		                          spacing=0)  # ,spacing=5, size=(1000,100))
		self.box_row2 = BoxLayout(orientation='horizontal', pos_hint={"y": 0.3})
		self.box_row3 = BoxLayout(orientation='horizontal', pos_hint={"y": 0.0})

		""" row 1 """
        #Logger.critical("whatever = {}".format(thing))
		self.txt = TextInput(hint_text='Please write your sentence here...', size_hint=(.9, .4))
		self.btn_clear = Button(text='Clear All', on_press=self.clearText, size_hint=(.1, .4))
		""" row 2 """
		self.btn_example = Button(text='Example', on_press=self.example, size_hint=(.1, .3))
		self.btn_confirm = Button(text='Confirm Your Phrase', on_press=self.confirm, size_hint=(.1, .3))
		self.btn_predict = Button(text='Click to Predict', on_press=self.predict, size_hint=(.1, .3))
		""" row 3 """
		self.btn_quit = Button(text='QUIT', on_press=self.quit, size_hint=(.1, .3))

		""" add widgets to row 1 """
		self.box_row1.add_widget(self.txt)
		self.box_row1.add_widget(self.btn_clear)
		""" add widgets to row 2 """
		self.box_row2.add_widget(self.btn_example)
		self.box_row2.add_widget(self.btn_confirm)
		self.box_row2.add_widget(self.btn_predict)
		""" add widgets to row 3 """
		self.box_row3.add_widget(self.btn_quit)

		""" add rows to window """
		self.grid.add_widget(self.box_row1)
		self.grid.add_widget(self.box_row2)
		self.grid.add_widget(self.box_row3)

		return self.grid

	def clearText(self, instance):
		"""
		Clean up the input window
		:param instance: None for now
		:return: None
		"""
		self.txt.text = ''

	def confirm(self):
		"""
		Confirm Input
		:return: None
		"""
		test_list[0] = self.label.get()
		print(test_list[0])

	def example(self):
		"""
		Show example of usage on Console
		:return: None
		"""
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
		if (self.lr_v == 1 and self.lr_a == 1):
			pil_im = Image.open('/Users/apple/Desktop/CSE 156/final_cse156/happy.jpeg', 'r')
			print("\tYou are Happy")
		elif (self.lr_v == 1 and self.lr_a == 0):
			pil_im = Image.open('/Users/apple/Desktop/CSE 156/final_cse156/chilling.jpeg', 'r')
			print("\tYou are just Chilling")
		elif (self.lr_v == 0 and self.lr_a == 1):
			pil_im = Image.open('/Users/apple/Desktop/CSE 156/final_cse156/angry.jpeg', 'r')
			print("\tYou are really Displeased or Pissed")
		elif (self.lr_v == 0 and self.lr_a == 0):
			pil_im = Image.open('/Users/apple/Desktop/CSE 156/final_cse156/sad.jpeg', 'r')
			print("\tYou are Bored or you are Sad")
		pil_im.show()

	def predict(self):
		"""
		Predict the output based off user's input
		:return: None, but show up result
		"""
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
		if (self.lr_v == 1 and self.lr_a == 1):
			pil_im = Image.open('/Users/apple/Desktop/CSE 156/final_cse156/happy.jpeg', 'r')
			print("\tYou are Happy")
		elif (self.lr_v == 1 and self.lr_a == 0):
			pil_im = Image.open('/Users/apple/Desktop/CSE 156/final_cse156/chilling.jpeg', 'r')
			print("\tYou are just Chilling")
		elif (self.lr_v == 0 and self.lr_a == 1):
			pil_im = Image.open('/Users/apple/Desktop/CSE 156/final_cse156/angry.jpeg', 'r')
			print("\tYou are really Displeased or Pissed")
		elif (self.lr_v == 0 and self.lr_a == 0):
			pil_im = Image.open('/Users/apple/Desktop/CSE 156/final_cse156/sad.jpeg', 'r')
			print("\tYou are Bored or you are Sad")
		pil_im.show()

	def quit(self):
		"""
		Quit application
		:return: None
		"""
		self.App.get_running_app().stop()


#if __name__ == '__main__':
Application().run()
