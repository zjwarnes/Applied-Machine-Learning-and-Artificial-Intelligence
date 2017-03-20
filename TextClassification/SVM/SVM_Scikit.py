import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import re
import pickle
from sklearn import metrics, svm



def cleanConversation(inputFile):
    data = pd.read_csv(inputFile, sep=',')
    cleanr = re.compile('<.*?>')
    data['coversation'] = data['conversation'].apply(lambda x: (re.sub(cleanr, '', x).encode('utf-8')))
    return data


def sklearnstuff(train, val, kernel_type, sze):
    #vectorize data
    train = np.array_split(train, sze)[0]
    count_vect = CountVectorizer(stop_words='english', ngram_range=(0,6))
    X_train_counts = count_vect.fit_transform(train['conversation'])
    labels = train['category']
    
    print 'vectorized'

    #transform data
    tf_transformer = TfidfTransformer().fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    X_train_tf.shape
    print 'transformed'
    X_test_counts = count_vect.transform(val['conversation'])
    X_test_tf = tf_transformer.transform(X_test_counts)
    labelsTest = val['category'].values

    #fit data 
    print 'fitting'    
    clf = svm.SVC(verbose=True, C= 1000000000.0, gamma='auto', kernel=kernel_type).fit(X_train_tf, labels)
    print 'fitted'
    predicted = clf.predict(X_test_tf)

    #print np.mean(predicted == labelsTest)
    names = ['hockey', 'movies', 'nba', 'news', 'nfl', 'politics', 'soccer', 'worldnews']
    print(metrics.classification_report(labelsTest, predicted, target_names=names))

    #more testing, predicting
    '''
    testInput = cleanConversation("test_input.csv")
    finaltest = pd.concat([testInput['id'], testInput['conversation']], axis=1)
    X_realTest_counts = count_vect.transform(finaltest['conversation'])
    X_realtest_tf = tf_transformer.transform(X_realTest_counts)
    predicted = clf.predict(X_realtest_tf)

    f = open('outputTestLogReg.csv', 'w')
    f.write('id,category\n')
    for indx,i in enumerate(predicted):
        f.write(str(indx) + ',' + str(i) + '\n')
        
    print(metrics.classification_report(labelsTest, predicted, target_names=names))
    '''
    
def main():    
    train = pickle.load(open('trainset','rb'))
    print len(train)
    val = pickle.load(open('valset','rb'))
    print len(val)

    kernel = 'rbf'
    fractionOfData = 100 # 1/100th of the total data
    sklearnstuff(train, val, kernel, fractionOfData)

if __name__ == "__main__":
    main()

