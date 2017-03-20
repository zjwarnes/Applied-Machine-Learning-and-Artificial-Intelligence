import pandas as pd
from pandas import DataFrame 
import numpy as np 
import re
import string
import sklearn.feature_extraction.text as text
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.naive_bayes import MultinomialNB
import pickle
import random
from collections import defaultdict
from itertools import islice
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfTransformer
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from sklearn import metrics, svm
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import operator
import math


stopwords = (stopwords.words('english'))
print stopwords
punct = list(string.punctuation)
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
remove = stopwords + punct + ["'s"]

#remove all the html tags
def cleanConversation(inputFile):
	data = pd.read_csv(inputFile, sep=',')
	cleanr = re.compile('<.*?>')
	data['conversation'] = data['conversation'].apply(lambda x: (re.sub(cleanr, '', x).encode('utf-8')))
	return data

#read the data
def readData():
	trainInputFile = "train_input.csv"
	trainOutputFile = "train_output.csv"
	testInput = "test_input.csv"
	testPredictRandom = "test_predict_random.csv"

	trainInput = cleanConversation(trainInputFile)
	trainOutput = pd.read_csv(trainOutputFile, sep=',')

	df = pd.concat([trainInput['id'], trainOutput['category'], trainInput['conversation']], axis=1)
	
	return df

def splitData(df):
	print len(df)
	train, validate = np.split(df.sample(frac=1, random_state=0), [int(.8*len(df))])
	print len(train['category'])
	print len(validate)

	trainFile = open('trainSet', 'wb')
	pickle.dump(train, trainFile)

	valFile = open('valSet', 'wb')
	pickle.dump(validate, valFile)


	return train, validate

#create a frequency table with all values
def freqTable(train, labels):

	dictCategories = {'hockey':1, 'movies':2, 'nba':3, 'news':4, 'nfl':5, 'politics':6, 'soccer':7, 'worldnews':8}

	maxVal = 20000
	convoList = train['conversation'].tolist()[0:maxVal]
	categoryList = train['category'].tolist()[0:maxVal]
	freq_tbl = pd.DataFrame([])
	stemmer = PorterStemmer()

	#splitConvos = [[w for w in convo.split() if w.lower() not in remove]
    #     for convo in convoList]
	splitConvos = [[stemmer.stem(w) for w in convo.split() if w.lower() not in remove]
         for convo in convoList]


	print len(splitConvos)

	listToAppend = []

	for idx, t in enumerate(splitConvos):
		print idx
		if idx == maxVal:
			break

		vocab = set(t)


		d = pd.Series({ v : t.count(v) for v in vocab})
		if labels == True:
			d['!label!'] = dictCategories[str(categoryList[idx])]
		else:
			d['!label!'] = 0

		listToAppend.append(d)
		
	freq_tbl = freq_tbl.append(listToAppend, ignore_index=True)

	columnList = list(freq_tbl)
	columnList.remove('!label!')


	freq_tbl = freq_tbl.fillna(0)
	freq_tbl.drop([col for col, val in freq_tbl.sum().iteritems() if val <= 20], axis=1, inplace=True)
	freq_tbl.drop([col for col, val in freq_tbl.sum().iteritems() if (val >= maxVal/3 and col != '!label!')], axis=1, inplace=True)

	#newFreqTable = freq_tbl.loc[:, (freq_tbl.sum(axis=0) > 1000)]

	#print freq_tbl
	#print newFreqTable

	#frequencyFile = open('freqTable', 'wb')
	#pickle.dump(freq_tbl, frequencyFile)
	print freq_tbl
	return freq_tbl



#create the probabiltiy matrices over the frequency table
def train_data(frequency_table):
	print frequency_table
	#print frequency_table['!label!']

	frequencies = frequency_table.iloc[:, 1:]

	labels = frequency_table.iloc[:, 0].values

	vocab = list(frequencies.columns.values)

	#d = {'hockey':1, 'movies':2, 'nba':3, 'news':4, 'nfl':5, 'politics':6, 'soccer':7, 'worldnews':8}

	hockey = pd.DataFrame([])
	movies = pd.DataFrame([])
	nba = pd.DataFrame([])
	news = pd.DataFrame([])
	nfl = pd.DataFrame([])
	politics = pd.DataFrame([])
	soccer = pd.DataFrame([])
	worldnews = pd.DataFrame([])

	print 'here'
	for idx, row in frequencies.iterrows():
		if labels[idx] == 1.0:
			hockey = hockey.append(row)
		elif labels[idx] == 2.0:
			movies = movies.append(row)
		elif labels[idx] ==  3.0:
			nba = nba.append(row)
		elif labels[idx] == 4.0:
			news = news.append(row)
		elif labels[idx] == 5.0:
			nfl = nfl.append(row)
		elif labels[idx] == 6.0:
			politics = politics.append(row)
		elif labels[idx] == 7.0:
			soccer = soccer.append(row)
		elif labels[idx] == 8.0:
			worldnews = worldnews.append(row)
		else:
			print labels[idx]
			print "WTF"

	print 'here2'
	hocProb, movProb, nbaProb, newsProb, nflProb, polProb, socProb, wnProb = {}, {}, {}, {}, {}, {}, {}, {}

	hocWordCount = sum([word for word in hockey.sum()])
	movWordCount = sum([word for word in movies.sum()])
	nbaWordCount = sum([word for word in nba.sum()])
	newsWordCount = sum([word for word in news.sum()])
	nflWordCount = sum([word for word in nfl.sum()])
	polWordCount = sum([word for word in politics.sum()])
	socWordCount = sum([word for word in soccer.sum()])
	wnWordCount = sum([word for word in worldnews.sum()])

	alpha = 1
	print 'here3'
	for word in vocab:
		hockeyOcc = hockey[word].sum()
		moviesOcc = movies[word].sum()
		nbaOcc = nba[word].sum()
		newsOcc = news[word].sum()
		nflOcc = nfl[word].sum()
		politicsOcc = politics[word].sum()
		soccerOcc = soccer[word].sum()
		worldNewsOcc = worldnews[word].sum()
		

		probHockey = math.log((hockeyOcc + alpha) / (hocWordCount + len(vocab)))
		probMovies = math.log((moviesOcc + alpha) / (movWordCount + len(vocab)))
		probNBA = math.log((nbaOcc + alpha) / (nbaWordCount + len(vocab)))
		probNews = math.log((newsOcc + alpha) / (newsWordCount + len(vocab)))
		probNfl = math.log((nflOcc + alpha) / (nflWordCount + len(vocab)))
		probPolitics = math.log((politicsOcc + alpha) / (polWordCount + len(vocab)))
		probSoccer = math.log((soccerOcc + alpha) / (socWordCount + len(vocab)))
		probWorldNews = math.log((worldNewsOcc + alpha) / (wnWordCount + len(vocab)))

		hocProb[word] = probHockey
		movProb[word] = probMovies
		nbaProb[word] = probNBA
		newsProb[word] = probNews
		nflProb[word] = probNfl
		polProb[word] = probPolitics
		socProb[word] = probSoccer
		wnProb[word] = probWorldNews


	print 'hockey: '
	maxHok = max(hocProb.iteritems(), key=operator.itemgetter(1))[0] 
	print maxHok
	print hocProb[str(maxHok)]
	print 
	print 'movies: '
	maxMov = str(max(movProb.iteritems(), key=operator.itemgetter(1))[0])
	print maxMov
	print movProb[maxMov]
	print 
	print 'nba: '
	maxNBA = str(max(nbaProb.iteritems(), key=operator.itemgetter(1))[0])
	print maxNBA
	print nbaProb[maxNBA]
	print 
	print 'news: '
	maxNews = str(max(newsProb.iteritems(), key=operator.itemgetter(1))[0])
	print maxNews
	print newsProb[maxNews]
	print 
	print 'nfl: '
	maxNfl = str(max(nflProb.iteritems(), key=operator.itemgetter(1))[0])
	print maxNfl
	print nflProb[maxNfl]
	print 
	print 'politics: '
	maxPol = str(max(polProb.iteritems(), key=operator.itemgetter(1))[0])
	print maxPol
	print polProb[maxPol]
	print 
	print 'soccer: '
	maxSoc = str(max(socProb.iteritems(), key=operator.itemgetter(1))[0])
	print maxSoc
	print socProb[maxSoc]
	print 
	print 'world news: '
	maxWN = str(max(wnProb.iteritems(), key=operator.itemgetter(1))[0])
	print maxWN
	print wnProb[maxWN]
	print 
	return hocProb, movProb, nbaProb, newsProb, nflProb, polProb, socProb, wnProb


#running the predict code over each predictor
def predict(validate, hocProb, movProb, nbaProb, newsProb, nflProb, polProb, socProb, wnProb):
	validate['prediction'] = 0

	#validate['prediction'] = list(map(predict_conversation,validate['conversation']))

	predictor = lambda string: predict_conversation(string,  hocProb, movProb, nbaProb, newsProb, nflProb, polProb, socProb, wnProb)
	validate['prediction'] = validate['conversation'].apply(predictor)
	return validate

## calculating probability of each category for each sample
def predict_conversation(convo, hocProb, movProb, nbaProb, newsProb, nflProb, polProb, socProb, wnProb):
	stemmer = PorterStemmer()

	#convoSplit = [w for w in convo.split() if w.lower() not in remove]
	convoSplit = [stemmer.stem(w) for w in convo.split() if w.lower() not in remove]

	vocab = set(convoSplit)

	d = pd.Series({ v : convoSplit.count(v) for v in vocab})
	
		
	freq_tbl = d

	freq_tbl = freq_tbl.fillna(0)

	words = freq_tbl.index.tolist()

	hockeyProbability = [0, 'hockey']
	movProbability = [0, 'movies']
	nbaProbability = [0, 'nba']
	newsProbability = [0, 'news']
	nflProbability = [0, 'nfl']
	polProbability = [0, 'politics']
	socProbability = [0, 'soccer']
	wnProbability = [0, 'worldnews']

	for wrd in words:

		if wrd in hocProb:
			hockeyProbability[0] = hockeyProbability[0] + hocProb[wrd]

		if wrd in movProb:
			movProbability[0] =  movProbability[0] + movProb[wrd]

		if wrd in nbaProb:
			nbaProbability[0] = nbaProbability[0] + nbaProb[wrd]

		if wrd in newsProb:
			newsProbability[0] = newsProbability[0] + newsProb[wrd]

		if wrd in nflProb:
			nflProbability[0] = nflProbability[0] + nflProb[wrd]

		if wrd in polProb:
			polProbability[0] = polProbability[0] + polProb[wrd]

		if wrd in socProb:
			socProbability[0] = socProbability[0] + socProb[wrd]

		if wrd in wnProb:
			wnProbability[0] = wnProbability[0] + wnProb[wrd]

	probabilities = [hockeyProbability, movProbability, nbaProbability, newsProbability, nflProbability, polProbability, socProbability, wnProbability]
	
	maxProbability = max(probabilities, key=lambda x: x[0])
	d = {'hockey':1, 'movies':2, 'nba':3, 'news':4, 'nfl':5, 'politics':6, 'soccer':7, 'worldnews':8}

	return maxProbability[1]

##Printing the results over the validation set
def testError(predictedSet):
	print type(predictedSet['category'])
	print type(predictedSet['prediction'])
	predictedSet['eqality'] = np.where(predictedSet['category'] == predictedSet['prediction'], True, False)
	print predictedSet

	countTrues = predictedSet['eqality'][predictedSet['eqality'] == True].count()
	countFalse = predictedSet['eqality'][predictedSet['eqality'] == False].count()

	print countTrues
	print countFalse

	accuracy = float(countTrues) / (countTrues + countFalse)

	names = ['hockey', 'movies', 'nba', 'news', 'nfl', 'politics', 'soccer', 'worldnews']


	print(metrics.classification_report(predictedSet['prediction'], predictedSet['category'], target_names=names))



	print accuracy

def compareEquality(firstString, secondString):
	return firstString == secondString


#Vectorizing and implementing the SKLearn classifiers
def sklearnstuff(train, val):
	print train
	print val

	train = np.array_split(train, 2)[0]
	print len(train)

	count_vect = CountVectorizer(stop_words='english', ngram_range=(0,6))
	X_train_counts = count_vect.fit_transform(train['conversation'])
	labels = train['category']
	
	print 'vectorized'


	tf_transformer = TfidfTransformer().fit(X_train_counts)
	X_train_tf = tf_transformer.transform(X_train_counts)
	X_train_tf.shape
	print 'transformed'
	X_test_counts = count_vect.transform(val['conversation'])
	X_test_tf = tf_transformer.transform(X_test_counts)

	labelsTest = val['category'].values

	print 'fitting'
	#clf = MultinomialNB().fit(X_train_tf, labels)
	#clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3,  shuffle=False, random_state=42, verbose=True).fit(X_train_tf, labels)
	#clf = RandomForestClassifier(min_samples_leaf=2, verbose=True).fit(X_train_tf, labels)
	#clf = OneVsRestClassifier(SVC(kernel='linear', probability=True, class_weight='auto')).fit(X_train_tf, labels)
	clf = LogisticRegression(solver='sag', max_iter=200, random_state=42, verbose=True, class_weight="balanced").fit(X_train_tf, labels)
	#

	#clf = svm.SVC(verbose=True, C=1000000.0, gamma='auto', kernel='rbf').fit(X_train_tf, labels)

	print 'fitted'
	predicted = clf.predict(X_test_tf)


	print np.mean(predicted == labelsTest)

	names = ['hockey', 'movies', 'nba', 'news', 'nfl', 'politics', 'soccer', 'worldnews']

	print(metrics.classification_report(labelsTest, predicted, target_names=names))

	entity.ent

	testInput = cleanConversation("test_input.csv")
	finaltest = pd.concat([testInput['id'], testInput['conversation']], axis=1)

	X_realTest_counts = count_vect.transform(finaltest['conversation'])
	X_realtest_tf = tf_transformer.transform(X_realTest_counts)
	predicted = clf.predict(X_realtest_tf)

	print predicted
	print len(predicted)

	f = open('outputTestLogReg.csv', 'w')
	f.write('id,category\n')

	for indx,i in enumerate(predicted):
		f.write(str(indx) + ',' + str(i) + '\n')
		print (indx, i)


def main():
	df = readData()
	train, validate = splitData(df)
	train = pickle.load(open('trainSet', 'rb'))
	#print len(train)
	#validate = pickle.load(open('valSet', 'rb'))
	#print len(validate)

	##Comment or don't comment if you want to work on SKLearn implementations
	#sklearnstuff(train, validate)
	

	frequencyTable = freqTable(train, True)

	print frequencyTable

	hocProb, movProb, nbaProb, newsProb, nflProb, polProb, socProb, wnProb = train_data(frequencyTable)
	
	validatePredictions = predict(validate, hocProb, movProb, nbaProb, newsProb, nflProb, polProb, socProb, wnProb)
	testError(validatePredictions)


if __name__ == '__main__':
   main()