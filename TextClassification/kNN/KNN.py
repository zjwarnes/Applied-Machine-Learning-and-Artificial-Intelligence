import pandas as pd 
import numpy as np 
import scipy
import nltk 
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re 
import operator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
ps = PorterStemmer()
wl = WordNetLemmatizer()

stop_words = set(stopwords.words("english"))

np.set_printoptions(threshold=np.nan)

def cleanhtml(text):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', text)
  return cleantext

#############################################
# PoS tagging

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
        
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
        
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
        
    elif treebank_tag.startswith('R'):
        return wordnet.ADV  
    else:
        return wordnet.NOUN

#############################################
#imports data 

train_input_data = pd.read_csv("train_input.csv")
train_output_data = pd.read_csv("train_output.csv")
train_output_data2 = pd.read_csv("train_output.csv")
train_output_data = train_output_data.head(27000)
train_input_data = train_input_data.head(27000)
test_output_data = pd.read_csv("train_output.csv")
test_output_data = test_output_data.tail(3000)

test_input_data = pd.read_csv("train_input.csv")
test_input_data = test_input_data.tail(3000)
test_input_data.reset_index(drop=True)

#print train_input_data
#print train_output_data['category'][2]
train_output_data.reset_index(drop=True)

train_full_data = pd.merge(train_input_data, train_output_data, on="id")
test_full_data = test_input_data

print train_full_data.shape

#tokenize and clean train data 
for index, row in train_full_data.iterrows():
  
    temporary_string = row['conversation']
    
    temporary_string2 = cleanhtml(temporary_string)

    tokenized_conversation = word_tokenize(temporary_string2)


    for i, w in enumerate(tokenized_conversation):
         tokenized_conversation[i] = ps.stem(w)

    
    temporary_string2 = " ".join(tokenized_conversation)


    train_full_data.set_value(index, 'conversation', temporary_string2)

#tokenize and clean valiation data
for index, row in test_full_data.iterrows():

    temporary_string = row['conversation']

    temporary_string2 = cleanhtml(temporary_string)
  

    tokenized_conversation = word_tokenize(temporary_string2)
  
    for i, w in enumerate(tokenized_conversation):
         tokenized_conversation[i] = ps.stem(w)
    
    temporary_string2 = " ".join(tokenized_conversation)


    test_full_data.set_value(index, 'conversation', temporary_string2)


corpus = train_full_data['conversation']


#Function to do TfIDF processing and normalizing of TruncatedSVD decomposition 
def tfidf_processor(X_train_set, X_test_set, features=0):
    vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase='true')
    X_train_set_tfidf = vectorizer.fit_transform(X_train_set)
    X_test_set_tfidf = vectorizer.transform(X_test_set)
    pca = PCA(n_components=25)
    nmf = NMF(n_components = 20)
    svd = TruncatedSVD(n_components=25)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    #transforms to TruncatedSVD 
    if features > 0:
        relevant_Features = np.argsort(vectorizer.idf_)[:features]
        X_train_set_tfidf = X_train_set_tfidf[:, relevant_Features]
        X_test_set_tfidf = X_test_set_tfidf[:,relevant_Features]
        X_train_set_SVD = lsa.fit_transform(X_train_set_tfidf)
        X_test_set_SVD = lsa.transform(X_test_set_tfidf)
        
  
    return(X_train_set_SVD, X_test_set_SVD)
    
#calculates distances
def calculate_Distance(a,b):
    #dist = scipy.spatial.distance.cityblock(a,b)
    #dist = np.linalg.norm(a-b)
    #dist = scipy.spatial.distance.minkowski(a,b,3)
    dist = scipy.spatial.distance.cosine(a,b)
    return dist

#calculates nearest neighbors
def calculateNearestNeighbors(X_train_set, X_test_Vector, k):
    distances = []
    for counter, x in enumerate(X_train_set):

        dist = calculate_Distance(X_test_Vector, x)
        distances.append((dist, counter))

    distances.sort(key=operator.itemgetter(0))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][1])
    return neighbors

print train_output_data

def outputValues(X_train_set, X_test_Vector): 
    print X_train_set.shape
    output_array = []
    for count,x in enumerate(X_test_Vector):
        nearest_neighbors = calculateNearestNeighbors(X_train_set, x, 7)
       # print nearest_neighbors
        d = {'hockey':0, 'movies':0, 'nba':0, 'news':0, 'nfl':0, 'politics':0, 'soccer':0, 'worldnews':0}
        for n in nearest_neighbors: 
        
            if train_output_data['category'][n] == 'hockey':
                d['hockey']+=1
            elif train_output_data['category'][n] == 'movies':
                d['movies']+=1
            elif train_output_data['category'][n] == 'nba':
                d['nba']+=1
            elif train_output_data['category'][n] == 'news':
                d['news']+=1
            elif train_output_data['category'][n] == 'nfl':
                d['nfl']+=1
            elif train_output_data['category'][n] == 'politics':
                d['politics']+=1
            elif train_output_data['category'][n] == 'soccer':
                d['soccer']+=1
            elif train_output_data['category'][n] == 'worldnews':
                d['worldnews']+=1
        category = max(d,key=d.get)
        row = [count, category]
        output_array.append(row)
   
    return output_array

print train_full_data['conversation'].shape
x_train_set, x_test_set = tfidf_processor(train_full_data['conversation'], test_full_data['conversation'],550)

print x_train_set.shape

final_output_array = test_output_data['category']

print"This is the final output array"
print final_output_array



array = outputValues(x_train_set, x_test_set)

totalNumber=0
totalCorrect=0

############################
total_number_NBA=0
total_correct_NBA=0
total_incorrect_NBA_NFL=0
total_incorrect_NBA_HOCKEY=0
total_incorrect_NBA_MOVIES=0
total_incorrect_NBA_NEWS=0
total_incorrect_NBA_WORLDNEWS=0
total_incorrect_NBA_POLITICS=0
total_incorrect_NBA_SOCCER=0
############################

############################
total_number_NFL=0
total_correct_NFL=0
total_incorrect_NFL_HOCKEY=0
total_incorrect_NFL_MOVIES=0
total_incorrect_NFL_NEWS=0
total_incorrect_NFL_WORLDNEWS=0
total_incorrect_NFL_POLITICS=0
total_incorrect_NFL_SOCCER=0
total_incorrect_NFL_NBA=0
############################


############################
total_number_HOCKEY=0
total_correct_HOCKEY=0
total_incorrect_HOCKEY_MOVIES=0
total_incorrect_HOCKEY_NEWS=0
total_incorrect_HOCKEY_WORLDNEWS=0
total_incorrect_HOCKEY_POLITICS=0
total_incorrect_HOCKEY_SOCCER=0
total_incorrect_HOCKEY_NBA=0
total_incorrect_HOCKEY_NFL=0
############################

############################
total_number_MOVIES=0
total_correct_MOVIES=0
total_incorrect_MOVIES_NEWS=0
total_incorrect_MOVIES_WORLDNEWS=0
total_incorrect_MOVIES_POLITICS=0
total_incorrect_MOVIES_SOCCER=0
total_incorrect_MOVIES_NBA=0
total_incorrect_MOVIES_NFL=0
total_incorrect_MOVIES_HOCKEY=0
############################

############################
total_number_NEWS=0
total_correct_NEWS=0
total_incorrect_NEWS_WORLDNEWS=0
total_incorrect_NEWS_POLITICS=0
total_incorrect_NEWS_SOCCER=0
total_incorrect_NEWS_NBA=0
total_incorrect_NEWS_NFL=0
total_incorrect_NEWS_HOCKEY=0
total_incorrect_NEWS_MOVIES=0
############################

############################
total_number_WORLDNEWS=0
total_correct_WORLDNEWS=0
total_incorrect_WORLDNEWS_POLITICS=0
total_incorrect_WORLDNEWS_SOCCER=0
total_incorrect_WORLDNEWS_NBA=0
total_incorrect_WORLDNEWS_NFL=0
total_incorrect_WORLDNEWS_HOCKEY=0
total_incorrect_WORLDNEWS_MOVIES=0
total_incorrect_WORLDNEWS_NEWS=0
############################


############################
total_number_POLITICS=0
total_correct_POLITICS=0
total_incorrect_POLITICS_WORLDNEWS=0
total_incorrect_POLITICS_SOCCER=0
total_incorrect_POLITICS_NBA=0
total_incorrect_POLITICS_NFL=0
total_incorrect_POLITICS_HOCKEY=0
total_incorrect_POLITICS_MOVIES=0
total_incorrect_POLITICS_NEWS=0
############################

############################
total_number_SOCCER=0
total_correct_SOCCER=0
total_incorrect_SOCCER_WORLDNEWS=0
total_incorrect_SOCCER_POLITICS=0
total_incorrect_SOCCER_NBA=0
total_incorrect_SOCCER_NFL=0
total_incorrect_SOCCER_HOCKEY=0
total_incorrect_SOCCER_MOVIES=0
total_incorrect_SOCCER_NEWS=0
############################

# print type(array[1][1])
for count, value in enumerate(final_output_array): 
    totalNumber +=1

    if (value == 'nba'):
        total_number_NBA +=1
        if (array[count][1] == value):
            total_correct_NBA+=1
            totalCorrect+=1
        elif (array[count][1] == 'nfl'):
            total_incorrect_NBA_NFL+=1
        elif (array[count][1] == 'hockey'):
            total_incorrect_NBA_HOCKEY+=1
        elif (array[count][1] == 'movies'):
            total_incorrect_NBA_MOVIES+=1
        elif (array[count][1] == 'news'):
            total_incorrect_NBA_NEWS+=1
        elif (array[count][1] == 'worldnews'):
            total_incorrect_NBA_WORLDNEWS+=1
        elif (array[count][1] == 'politics'):
            total_incorrect_NBA_POLITICS+=1
        elif (array[count][1] == 'soccer'):
            total_incorrect_NBA_SOCCER+=1
        
    elif (value == 'nfl'):
        total_number_NFL +=1
        if (array[count][1] == value):
            total_correct_NFL+=1
            totalCorrect+=1
        elif (array[count][1] == 'nba'):
            total_incorrect_NFL_NBA+=1
        elif (array[count][1] == 'hockey'):
            total_incorrect_NFL_HOCKEY+=1
        elif (array[count][1] == 'movies'):
            total_incorrect_NFL_MOVIES+=1
        elif (array[count][1] == 'news'):
            total_incorrect_NFL_NEWS+=1
        elif (array[count][1] == 'worldnews'):
            total_incorrect_NFL_WORLDNEWS+=1
        elif (array[count][1] == 'politics'):
            total_incorrect_NFL_POLITICS+=1
        elif (array[count][1] == 'soccer'):
            total_incorrect_NFL_SOCCER+=1
        
    
    elif (value == 'hockey'):
        total_number_HOCKEY +=1
        if (array[count][1] == value):
            total_correct_HOCKEY+=1
            totalCorrect+=1
        elif (array[count][1] == 'nba'):
            total_incorrect_HOCKEY_NBA+=1
        elif (array[count][1] == 'nfl'):
            total_incorrect_HOCKEY_NFL+=1
        elif (array[count][1] == 'movies'):
            total_incorrect_HOCKEY_MOVIES+=1
        elif (array[count][1] == 'news'):
            total_incorrect_HOCKEY_NEWS+=1
        elif (array[count][1] == 'worldnews'):
            total_incorrect_HOCKEY_WORLDNEWS+=1
        elif (array[count][1] == 'politics'):
            total_incorrect_HOCKEY_POLITICS+=1
        elif (array[count][1] == 'soccer'):
            total_incorrect_HOCKEY_SOCCER+=1
    
    elif (value == 'news'):
        total_number_NEWS +=1
        if (array[count][1] == value):
            total_correct_NEWS+=1
            totalCorrect+=1
        elif (array[count][1] == 'nba'):
            total_incorrect_NEWS_NBA+=1
        elif (array[count][1] == 'nfl'):
            total_incorrect_NEWS_NFL+=1
        elif (array[count][1] == 'movies'):
            total_incorrect_NEWS_MOVIES+=1
        elif (array[count][1] == 'hockey'):
            total_incorrect_NEWS_HOCKEY+=1
        elif (array[count][1] == 'worldnews'):
            total_incorrect_NEWS_WORLDNEWS+=1
        elif (array[count][1] == 'politics'):
            total_incorrect_NEWS_POLITICS+=1
        elif (array[count][1] == 'soccer'):
            total_incorrect_NEWS_SOCCER+=1

    elif (value == 'worldnews'):
        total_number_WORLDNEWS +=1
        if (array[count][1] == value):
            total_correct_WORLDNEWS+=1
            totalCorrect+=1
        elif (array[count][1] == 'nba'):
            total_incorrect_WORLDNEWS_NBA+=1
        elif (array[count][1] == 'nfl'):
            total_incorrect_WORLDNEWS_NFL+=1
        elif (array[count][1] == 'movies'):
            total_incorrect_WORLDNEWS_MOVIES+=1
        elif (array[count][1] == 'hockey'):
            total_incorrect_WORLDNEWS_HOCKEY+=1
        elif (array[count][1] == 'news'):
            total_incorrect_WORLDNEWS_NEWS+=1
        elif (array[count][1] == 'politics'):
            total_incorrect_WORLDNEWS_POLITICS+=1
        elif (array[count][1] == 'soccer'):
            total_incorrect_WORLDNEWS_SOCCER+=1

    elif (value == 'politics'):
        total_number_POLITICS +=1
        if (array[count][1] == value):
            total_correct_POLITICS+=1
            totalCorrect+=1
        elif (array[count][1] == 'nba'):
            total_incorrect_POLITICS_NBA+=1
        elif (array[count][1] == 'nfl'):
            total_incorrect_POLITICS_NFL+=1
        elif (array[count][1] == 'movies'):
            total_incorrect_POLITICS_MOVIES+=1
        elif (array[count][1] == 'hockey'):
            total_incorrect_POLITICS_HOCKEY+=1
        elif (array[count][1] == 'news'):
            total_incorrect_POLITICS_NEWS+=1
        elif (array[count][1] == 'worldnews'):
            total_incorrect_POLITICS_WORLDNEWS+=1
        elif (array[count][1] == 'soccer'):
            total_incorrect_POLITICS_SOCCER+=1

    elif (value == 'soccer'):
        total_number_SOCCER +=1
        if (array[count][1] == value):
            total_correct_SOCCER+=1
            totalCorrect+=1
        elif (array[count][1] == 'nba'):
            total_incorrect_SOCCER_NBA+=1
        elif (array[count][1] == 'nfl'):
            total_incorrect_SOCCER_NFL+=1
        elif (array[count][1] == 'movies'):
            total_incorrect_SOCCER_MOVIES+=1
        elif (array[count][1] == 'hockey'):
            total_incorrect_SOCCER_HOCKEY+=1
        elif (array[count][1] == 'news'):
            total_incorrect_SOCCER_NEWS+=1
        elif (array[count][1] == 'worldnews'):
            total_incorrect_SOCCER_WORLDNEWS+=1
        elif (array[count][1] == 'politics'):
            total_incorrect_SOCCER_POLITICS+=1

    elif (value == 'movies'):
        total_number_MOVIES +=1
        if (array[count][1] == value):
            total_correct_MOVIES+=1
            totalCorrect+=1
        elif (array[count][1] == 'nba'):
            total_incorrect_MOVIES_NBA+=1
        elif (array[count][1] == 'nfl'):
            total_incorrect_MOVIES_NFL+=1
        elif (array[count][1] == 'soccer'):
            total_incorrect_MOVIES_SOCCER+=1
        elif (array[count][1] == 'hockey'):
            total_incorrect_MOVIES_HOCKEY+=1
        elif (array[count][1] == 'news'):
            total_incorrect_MOVIES_NEWS+=1
        elif (array[count][1] == 'worldnews'):
            total_incorrect_MOVIES_WORLDNEWS+=1
        elif (array[count][1] == 'politics'):
            total_incorrect_MOVIES_POLITICS+=1

   ## Prints all information 

print "Total Correct"
print totalCorrect
print "Total Number"
print totalNumber

print "Overall Accuracy"
print (totalCorrect/float(totalNumber))
print " "

print "NBA Accuracy"
print (total_correct_NBA/float(total_number_NBA))
print " "

print "NFL Accuracy"
print (total_correct_NFL/float(total_number_NFL))
print " "

print "HOCKEY Accuracy"
print (total_correct_HOCKEY/float(total_number_HOCKEY))
print " "

print "MOVIES Accuracy"
print (total_correct_MOVIES/float(total_number_MOVIES))
print " "

print "NEWS Accuracy"
print (total_correct_NEWS/float(total_number_NEWS))
print " "

print "WORLDNEWS Accuracy"
print (total_correct_WORLDNEWS/float(total_number_WORLDNEWS))
print " "

print "POLITICS Accuracy"
print (total_correct_POLITICS/float(total_number_POLITICS))
print " "

print "SOCCER Accuracy"
print (total_correct_SOCCER/float(total_number_SOCCER))
print " "

print "********CONFUSION MATRIX INFORMATION********"
print " "

print "total_incorrect_NBA_NFL= %d" %total_incorrect_NBA_NFL
print "total_incorrect_NBA_HOCKEY= %d" %total_incorrect_NBA_HOCKEY
print "total_incorrect_NBA_MOVIES= %d" %total_incorrect_NBA_MOVIES
print "total_incorrect_NBA_NEWS= %d" %total_incorrect_NBA_NEWS
print "total_incorrect_NBA_WORLDNEWS= %d" %total_incorrect_NBA_WORLDNEWS
print "total_incorrect_NBA_POLITICS= %d" %total_incorrect_NBA_POLITICS
print "total_incorrect_NBA_SOCCER= %d" %total_incorrect_NBA_SOCCER

print "***************"
print " "

print "total_incorrect_NFL_HOCKEY=%d" %total_incorrect_NFL_HOCKEY
print "total_incorrect_NFL_MOVIES=%d" %total_incorrect_NFL_MOVIES
print "total_incorrect_NFL_NEWS=%d" %total_incorrect_NFL_NEWS
print "total_incorrect_NFL_WORLDNEWS=%d" %total_incorrect_NFL_WORLDNEWS
print "total_incorrect_NFL_POLITICS=%d" %total_incorrect_NFL_POLITICS
print "total_incorrect_NFL_SOCCER=%d" %total_incorrect_NFL_SOCCER
print "total_incorrect_NFL_NBA=%d" %total_incorrect_NFL_NBA

print "***************"
print " "

print "total_incorrect_HOCKEY_MOVIES=%d" %total_incorrect_HOCKEY_MOVIES
print "total_incorrect_HOCKEY_NEWS=%d" %total_incorrect_HOCKEY_NEWS
print "total_incorrect_HOCKEY_WORLDNEWS=%d" %total_incorrect_HOCKEY_WORLDNEWS
print "total_incorrect_HOCKEY_POLITICS=%d" %total_incorrect_HOCKEY_POLITICS
print "total_incorrect_HOCKEY_SOCCER=%d" %total_incorrect_HOCKEY_SOCCER
print "total_incorrect_HOCKEY_NBA=%d" %total_incorrect_HOCKEY_NBA
print "total_incorrect_HOCKEY_NFL=%d" %total_incorrect_HOCKEY_NFL

print "***************"
print " "


print "total_incorrect_MOVIES_NEWS=%d" %total_incorrect_MOVIES_NEWS
print "total_incorrect_MOVIES_WORLDNEWS=%d" %total_incorrect_MOVIES_WORLDNEWS
print "total_incorrect_MOVIES_POLITICS=%d" %total_incorrect_MOVIES_POLITICS
print "total_incorrect_MOVIES_SOCCER=%d" %total_incorrect_MOVIES_SOCCER
print "total_incorrect_MOVIES_NBA=%d" %total_incorrect_MOVIES_NBA
print "total_incorrect_MOVIES_NFL=%d" %total_incorrect_MOVIES_NFL
print "total_incorrect_MOVIES_HOCKEY=%d" %total_incorrect_MOVIES_HOCKEY

print "***************"
print " "

print "total_incorrect_NEWS_WORLDNEWS=%d" %total_incorrect_NEWS_WORLDNEWS
print "total_incorrect_NEWS_POLITICS=%d" %total_incorrect_NEWS_POLITICS
print "total_incorrect_NEWS_SOCCER=%d" %total_incorrect_NEWS_SOCCER
print "total_incorrect_NEWS_NBA=%d" %total_incorrect_NEWS_NBA
print "total_incorrect_NEWS_NFL=%d" %total_incorrect_NEWS_NFL
print "total_incorrect_NEWS_HOCKEY=%d" %total_incorrect_NEWS_HOCKEY
print "total_incorrect_NEWS_MOVIES=%d" %total_incorrect_NEWS_MOVIES
            
print "***************"
print " "    

print "total_incorrect_WORLDNEWS_POLITICS=%d" %total_incorrect_WORLDNEWS_POLITICS
print "total_incorrect_WORLDNEWS_SOCCER=%d" %total_incorrect_WORLDNEWS_SOCCER
print "total_incorrect_WORLDNEWS_NBA=%d" %total_incorrect_WORLDNEWS_NBA
print "total_incorrect_WORLDNEWS_NFL=%d" %total_incorrect_WORLDNEWS_NFL
print "total_incorrect_WORLDNEWS_HOCKEY=%d" %total_incorrect_WORLDNEWS_HOCKEY
print "total_incorrect_WORLDNEWS_MOVIES=%d" %total_incorrect_WORLDNEWS_MOVIES
print "total_incorrect_WORLDNEWS_NEWS=%d" %total_incorrect_WORLDNEWS_NEWS

print "***************"
print " " 

print "total_incorrect_POLITICS_WORLDNEWS=%d" %total_incorrect_POLITICS_WORLDNEWS
print "total_incorrect_POLITICS_SOCCER=%d" %total_incorrect_POLITICS_SOCCER
print "total_incorrect_POLITICS_NBA=%d" %total_incorrect_POLITICS_NBA
print "total_incorrect_POLITICS_NFL=%d" %total_incorrect_POLITICS_NFL
print "total_incorrect_POLITICS_HOCKEY=%d" %total_incorrect_POLITICS_HOCKEY
print "total_incorrect_POLITICS_MOVIES=%d" %total_incorrect_POLITICS_MOVIES
print "total_incorrect_POLITICS_NEWS=%d" %total_incorrect_POLITICS_NEWS

print "***************"
print " " 

print "total_incorrect_SOCCER_WORLDNEWS=%d" %total_incorrect_SOCCER_WORLDNEWS
print "total_incorrect_SOCCER_POLITICS=%d" %total_incorrect_SOCCER_POLITICS
print "total_incorrect_SOCCER_NBA=%d" %total_incorrect_SOCCER_NBA
print "total_incorrect_SOCCER_NFL=%d" %total_incorrect_SOCCER_NFL
print "total_incorrect_SOCCER_HOCKEY=%d" %total_incorrect_SOCCER_HOCKEY
print "total_incorrect_SOCCER_MOVIES=%d" %total_incorrect_SOCCER_MOVIES
print "total_incorrect_SOCCER_NEWS=%d" %total_incorrect_SOCCER_NEWS





