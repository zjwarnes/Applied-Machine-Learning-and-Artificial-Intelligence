from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import random as rand
import csv
import parseTestData as ptd

#test data from coursera
def parseData():
    with open("ex2data2.txt", 'rb') as txtfile:
        reader = csv.reader(txtfile, delimiter = ',', quotechar='|' )
        count = 0
        arr = np.array([[]])
        for row in reader:
            if count == 0:
                arr = np.array([[float(row[0]),float(row[1]),float(row[2])]])
                count +=1 
            else:   
                arr = np.append(arr,np.array([[float(row[0]),float(row[1]),float(row[2])]]),axis = 0)
            
        txtfile.close()
        return arr
    
#as defined in slides
def exactMethod(X,Y):
    Xt = np.transpose(X)
    XtX = np.dot(Xt,X)
    XInv = np.linalg.inv(XtX)
    XtY = np.dot(Xt,Y)
    weights = np.dot(XInv, XtY)
    return weights

#assign random values to each weight between 0 and 1
def initializeWeights(size):
    w0 = np.array([[1.0]])
    for i in range(1,size):
        w0 = np.append(w0,np.array([rand.randint(0,10)/10]))
    return np.array([w0])

def normalize(col):
    mew = np.mean(col)
    vr = np.var(col)
    sd = np.sqrt(vr)
    if sd == 0:
        return np.ones(np.shape(col))
    return (col - mew)/sd

def scaleToHours(col):
    return col/3600 

#convert results back into string
def hoursIntoString(flt):
    hours = int(np.floor(flt))
    flt = flt - hours
    minutes = int(np.floor(flt*60))
    flt = flt*60 - minutes
    seconds = int(np.floor(flt*60))
    if hours < 10:
        hours = '0'+str(hours)
    if seconds < 0:
        seconds = '0'+str(seconds)
    if minutes < 0:
        minutes = '0'+str(minutes)
    return str(hours)+':'+str(minutes)+":"+str(seconds)  


def calcErr(X,y,wk):
    hx = np.dot(X,np.transpose(wk))
    diff = hx - y
    print np.dot(np.transpose((abs(diff/y)*100)),(abs(diff/y)*100))
    
    print np.dot(np.transpose(hx-y),hx-y)
    return np.average((abs(diff/y)*100))
    #return np.sqrt(np.power(abs(diff/y)*100,2))
 
def accuracy(X,y,wk):
    hx = np.dot(X,np.transpose(wk))
    dif = abs(hx-y)
    return np.average((dif/y)*100) 
 
#example data shown in lecture 2   
#X = np.transpose(np.array([[0.86,0.09,-0.85,0.87,-0.44,-0.43,-1.1,0.40,-0.96,0.17],[1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]]))
#Y = np.transpose(np.array([[2.49,0.83,-0.25,3.10,0.87,0.02,-0.12,1.81,-0.83,0.43]]))
def costFunction(X,y,weights, lbda):
    n = np.size(y, axis=0)
    reg = (lbda/(2*n))*np.dot(weights,np.transpose(weights))    
    hx = np.dot(X,np.transpose(weights))
    cost = (1/2/n)*np.dot(np.transpose(hx-y),(hx-y))
    cost += reg
    return cost[0][0]


def gradientDescent(X,y,w0,alpha,lbda):
    k = 1
    #helpful variables
    n = np.size(y)
    wk = w0
    
    #initialize first weight, ignoring regularization on bias term
    hx = np.dot(X,np.transpose(wk))
    biasGrad = wk[0] - alpha*(1/n)*np.dot(np.transpose(hx -y),X[:,1])
    wk1 = wk*(1-(alpha*lbda/n)) - alpha*np.dot(np.transpose(hx -y),X) 
    wk1[0,1] = biasGrad[1]
    
    weights = []
    while k < 50:
        #alpha = alpha*(1/(k*4))
        #track the weights for each iteration
        weights.append(wk)
        hx = np.dot(X,np.transpose(wk))
        
        # wk1 = reg + grad
        biasGrad = wk[0] - alpha*(1/n)*np.dot(np.transpose(hx -y),X[:,1])
        wk1 = wk*(1-alpha*lbda/n) - alpha*np.dot(np.transpose(hx -y),X)
        wk1[0,1] = biasGrad[1]
        wk = wk1
        k+=1
    
    #return the weights for each iteration
    return weights

#Randomize data, divide into cv and test, run gradient descent on test
#Calculate cost for each of the weights calculated against cv and test
#plot data against each other for training curve
def crossValidation(X,y,w0,k,alpha,lbda):
    #combine the two set to randomize rows first
    data = np.append(X, y, axis=1)
    np.random.shuffle(data)
    #useful values
    n = np.size(y)
    m = np.size(w0)
    
    #to average error over each iteration
    trainErr = []
    validErr = []
    acc = []
    for i in range(0,k):
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
        trainCurve = []
        cvCurve = []
        #returns a list of weights from gradient descent
        weights = gradientDescent(testSet[:,0:m], testSet[:,m:m+1], w0, alpha, lbda)
        for wk in weights:
            trainCurve.append(costFunction(testSet[:,0:m], testSet[:,m:m+1], wk, lbda))
            cvCurve.append(costFunction(cvSet[:,0:m], cvSet[:,m:m+1], wk, lbda))
        
        trainErr.append(costFunction(testSet[:,0:m], testSet[:,m:m+1], weights[48],lbda))
        validErr.append(costFunction(cvSet[:,0:m], cvSet[:,m:m+1], weights[48],lbda))
        #calculate accuracy from weights trained on test set, compared to cv set
        acc.append(accuracy(cvSet[:,0:m], cvSet[:,m:m+1], weights[48]))
        
        #plotCurve(cvCurve, trainCurve)
        
    print 'Mean train error over 5 iterations: ', np.mean(trainErr)
    print 'Mean valid error over 5 iterations: ', np.mean(validErr)
    print 'Accuracy over 5 iterations', np.mean(acc)
    print ''
    #weights at 48 is the optimal 
    return [np.mean(validErr), weights[48]]
    
def plotCurve(cvCurve, trainCurve):
    k = 50
    t = np.arange(1,50,1)    
    plt.plot(t, cvCurve,label = 'cv')
    plt.plot(t, trainCurve, label = 'train')
    plt.axis([0,k,0,1])
    plt.legend()
    plt.show() 
    
def linearReg():
    #take the following many data points or all within the file
    
    rows = 40000
    IDs = 50000
    fileName = 'Project1_data_formatted.csv'
    #load values into a dictionary
    dct = ptd.createRunnerDatabase(fileName, rows)
    
    #setup Hyper-parameters
    alpha = 0.00003#[0.001,0.003,0.01,0.03,0.1,0.3,1.0]
    kfold=5
    
    #TO run faster, switch the comments with the TWO values below
    lbda = [0.1,0.3,1.0,3.0,10]
    num = 5    
    #lbda = 1.0
    #num = 1

    #features: 1=age, 2= gender,  3 = times, 4 = years, 5 = improvementRatio, 6 = attendanceRatio
    features = [1,2,4,5]
    #generate data
    y = ptd.createYArray(dct, IDs, 2)
    X = ptd.createXArray(dct, IDs, features,0)
    
    #typ = 2 is times
    w0 = initializeWeights(np.size(X,axis=1))
    m = np.size(w0)
    y = scaleToHours(y)

    minError = 10000

    print 'Alpha: ', alpha
    #to check the degree of a feature
      
    #reload features to avoid problems with squaring normalized features
    X = ptd.createXArray(dct, IDs, features,0)
    X[:,1] = np.power(X[:,1],3)
            
    #normalize the features, after squaring
    for k in range(1,m):
        X[:,k] = normalize(X[:,k])
    
    for w in range(0,num):
        w0 = initializeWeights(np.size(X,axis=1))
        for j in range(len(lbda)): 
                  
            [runError, weights] = crossValidation(X, y, w0, kfold, alpha, lbda[j])    
            if runError < minError:
                minError = runError
                optWeights = weights
                optLambda = lbda[j]
    print "Results"
    print 'Min Error: ', minError
    print 'Optimal weights: ',optWeights
    print 'Optimal Lambda: ', optLambda
    
    X = ptd.createPredictionXArray(dct,IDs,features)
    X[:,1] = np.power(X[:,1],3)
        
    #normalize the features, after squaring
    for k in range(1,m):
        X[:,k] = normalize(X[:,k])
    
    prediction = np.dot(X,np.transpose(optWeights)) 
    
    with open('file.csv', 'wb+') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['PARTICIPANT_ID','Y2_REGRESSION'])
        count = 0
        for i in range(1,IDs):
            if dct.get(str(i)) == None:
                continue
            row=[]
            row.append(str(i))
            row.append(hoursIntoString(prediction[count,:]))
            writer.writerow(row)
            count+=1
        csvfile.close()    

#Uncomment me

#linearReg()






