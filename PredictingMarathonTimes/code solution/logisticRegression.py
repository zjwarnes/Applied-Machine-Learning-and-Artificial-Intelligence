from __future__ import division
import numpy as np
import random as rand
import matplotlib.pyplot as plt
import csv
import parseTestData as ptd

#calculates the sigmoid function, can pass in matrices
def sigmoid(z):
    denom = 1 + np.exp(-1*z)
    return 1/denom

#classify a example based on hypothesis
def predict(z, threshold):
    prob = sigmoid(z)
    if prob > threshold:
        return 1
    else: 
        return 0
#normalize a column
def normalize(col):
    mew = np.mean(col)
    vr = np.var(col)
    sd = np.sqrt(vr)
    return (col - mew)/sd

#print each of the means of measures listed in lectures
def printMeasures(validAcc, k):        
    measures = ['Accuracy', 'Precision', 'Recall', 'Speciificty', 'False Positive Rate', 'F1 Measure']   
    for j in range(len(measures)):
        print measures[j]
        #print 'Train: ', np.mean(trainAcc[1:k+1,j])
        print 'Valid: ', np.mean(validAcc[1:k+1,j])
    return      

#Plot the curves against each other
def plotCurve(cvCurve, trainCurve):    
    t = np.arange(1,50,1)    
    plt.plot(t, cvCurve,label = 'cv')
    plt.plot(t, trainCurve, label = 'train')
    plt.legend()
    plt.show()
    return 

def calcAcc(X,y,wk, threshold):
    z = np.dot(X,np.transpose(wk))
    prediction = sigmoid(z)
    
    #masking the prediction for a certain threshold
    prediction = prediction > threshold
    y = y > threshold
    
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(np.size(y)):
        if y[i][0] == True and prediction[i][0] == True:
            TP +=1
        elif y[i][0] == False and prediction[i][0] == True:
            FP+=1
        elif y[i][0] == True and prediction[i][0] == False:
            FN+=1
        else:
            TN+=1
    accuracy = (TP + TN)/(TP+FP+FN+TN)
    if TP == 0 and FP == 0:
        precision = 0
    else:
        precision = (TP/(TP+FP))
    
    if TP == 0 and FN == 0:
        recall = 0
    else:
        recall = (TP/(TP+FN))
    
    if FP == 0 and TN == 0:
        specificity = 0
        falsePosRate = 0
    else:
        specificity = (TN/(FP+TN))
        falsePosRate = (FP/(FP+TN))
    
    if precision == 0 and recall == 0:
        F1 = 0
    else:
        F1 = 2*(precision*recall)/(precision+recall)

    return np.array([[accuracy,precision,recall,specificity,falsePosRate,F1]])
    
    

#randomly initialize weights, w0 set to 1
def initializeWeights(size):
    w0 = np.array([1.0])
    for i in range(1,size):
        w0 = np.append(w0,np.array([rand.randint(0,10)/10]))
    return np.array([w0])

#calculate cost function for given weights
def costFunc(X, y, weights, lbda):
    n = np.size(y)

    #each weight excluding the bias term
    features = np.array(weights[1:])
    reg = (lbda/(2*n))* np.sum(features*features)
    z = np.dot(X,np.transpose(weights))
    
    cost = (1/n)*np.sum((-y)*np.log(sigmoid(z)) - (1-y)*np.log(1-sigmoid(z))) + reg    
    return cost
    
#calculate the gradient for given weights    
def gradient(X, y,  weights, lbda):
    n = np.size(y)
    z = np.dot(X,np.transpose(weights))
    reg = lbda * weights *(1/n)
    
    #add regularization to all of the weights
    grad = (1/n)*(np.dot((np.transpose(sigmoid(z)-y)),X)) + reg
    
    #remove regularization from bias term
    gradNoReg =  (1/n)*(np.dot((np.transpose(sigmoid(z)-y)),X))
    grad[:,0] = gradNoReg[:,0]
    return grad
    
def gradientDescent(X,y,w0,alpha, lbda):
    k=1
    #initialize first and second weights
    wk = w0
    wk1 = wk - alpha*gradient(X, y, wk, lbda)
    
    #weights tracks the wk values at each step
    weights = []
    while k <50: 
        weights.append(wk)
        wk1 = wk - alpha*gradient(X,y,wk,lbda)
        wk = wk1
        k+=1
    
    #return all of the weights to plot the training curves
    return weights
   
#Randomize all the data, split into cv and test sets
#Run gradient descent on the test set to get weights
#calculate cost at each step for train and cv sets and plot
def crossValidation(X,y,w0,k,alpha,lbda,threshold):
    #combine the two set to randomize rows first
    data = np.append(X, y, axis=1)
    np.random.shuffle(data)
    #useful values
    n = np.size(y)
    m = np.size(w0)
    
    trainAcc = np.array([[0,0,0,0,0,0]])
    validAcc = np.array([[0,0,0,0,0,0]])
    for i in range(0,k):
        #divide data into test and cv sets
        start = int(np.floor(i*n/k))
        stop = int(np.floor((i+1)*n/k))
        
        cvSet = data[start:stop,:]
        if i ==0:
            testSet = data[stop:n,:]
        elif i == k-1:
            testSet = data[0:start,:]
        else:
            t1 = data[0:start,:]
            t2 = data[stop:n,:]
            testSet = np.append(t1,t2,axis=0)
            
        #calculate cost at each step from
        #weights calculated using the training set
        trainCurve = []
        cvCurve = []
        weights = gradientDescent(testSet[:,0:m], testSet[:,m:m+1], w0, alpha, lbda)
        for wk in weights:
            trainCurve.append(costFunc(testSet[:,0:m], testSet[:,m:m+1], wk, lbda))
            cvCurve.append(costFunc(cvSet[:,0:m], cvSet[:,m:m+1], wk, lbda))
        
        #calculate the accuracy, precision, recall, specificity, falsePositive, F1 score
        trainAcc = np.append(trainAcc, calcAcc(testSet[:,0:m], testSet[:,m:m+1], weights[48], threshold),axis =0)
        validAcc = np.append(validAcc, calcAcc(cvSet[:,0:m], cvSet[:,m:m+1], weights[48], threshold),axis=0)
        
        #plotCurve(cvCurve, trainCurve)
    
    
    printMeasures(validAcc, k)
    return [validAcc, weights[k-1]]

def logisticReg():
    #take the following many data points or all within the file
    rows = 100000
    IDs = 40000
    fileName = 'Project1_data_formatted.csv'
    #load values into a dictionary
    dct = ptd.createRunnerDatabase(fileName, rows)
    
    #setup Hyper-parameters
    alpha = 1.0
    threshold = 0.6
    kfold=5
    
    #TO run faster, switch the comments with the TWO values below
    lbda = [0.1,0.3,1.0,3.0,10]
    num = 5    
    #lbda = 1.0
    #num = 1
    
    
    #features: 1=age, 2= gender, 3 = years, 4 = times, 5 = improvementRatio, 6 = attendanceRatio
    features = [1,2,6]
    #generate data, create X before y, since X can create negative dictionary entries
    X = ptd.createXArray(dct, IDs, features,1)
    y = ptd.createYArray(dct, IDs, 1)
    
    #typ = 2 is times
    w0 = initializeWeights(np.size(X,axis=1))
    m = np.size(w0)
    maxAcc = 0
    print 'Alpha: ', alpha
    X = ptd.createXArray(dct, IDs, features,0)
    
    #age deduced to perform best cube
    X[:,1] = np.power(X[:,1],3)
            
    #normalize the features, after squaring
    for k in range(1,m):
        X[:,k] = normalize(X[:,k])
    #X[:,1] = normalize(X[:,1])
            
    for w in range(0,num):
        w0 = initializeWeights(np.size(X,axis=1))    
        for j in range(len(lbda)):
            print 'Lambda: ', lbda[j]
            print '' 
            [validMeasure,weights]= crossValidation(X, y, w0, kfold, alpha, lbda[j], threshold)
            if np.mean(validMeasure[1:kfold+1,0]) > maxAcc:
                #'Accuracy', 'Precision', 'Recall', 'Speciificty', 'False Positive Rate', 'F1 Measure']   
                maxAcc = np.mean(validMeasure[1:kfold+1,0])
                optRecall = np.mean(validMeasure[1:kfold+1,2])
                optPrecision = np.mean(validMeasure[1:kfold+1,1])
                optWeights = weights
                optLambda = lbda[j]
                    
    print "Results"
    print 'Min Error: ', maxAcc
    print 'Optimal weights: ',optWeights
    print 'Optimal Lambda: ', optLambda
    print 'Optimal Recall:' , optRecall
    print "Optimal precision: ", optPrecision
    
    X = ptd.createPredictionXArray(dct,IDs,features)
    X[:,1] = np.power(X[:,1],3)
        
    #normalize the features, after squaring
    for k in range(1,m):
        X[:,k] = normalize(X[:,k])
    
    prediction = sigmoid(np.dot(X,np.transpose(optWeights))) > threshold 
    
    with open('file.csv', 'wb+') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['PARTICIPANT_ID','Y1_LOGISTIC'])
        count = 0
        for i in range(1,IDs):
            if dct.get(str(i)) == None:
                continue
            row=[]
            row.append(str(i))
            bol = prediction[count,:]
            if bol == True:
                row.append(1)
            else:
                row.append(0)
            writer.writerow(row)
            count+=1
        csvfile.close()  
    
  
#Uncomment ME
  
#logisticReg()


