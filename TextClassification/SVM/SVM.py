import numpy as np
from cvxpy import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle

class SVM():
    def __init__(self, iter_max = 10000, kernel_type = 'linear', C = 1.0, epsilon = 1.0e-6, sigma = 1.0, lambd = 1.0, norm = 1.0):
        self.kernels = {
            'linear': self.kernel_linear,
            'quadratic': self.kernel_quadratic,
            'gaussian': self.kernel_gaussian
        }
        self.iter_max = iter_max
        self.kernel_type = kernel_type
        self.sigma = sigma
        self.C = C
        self.norm = norm
        self.lambd = lambd
        self.epsilon = epsilon
    #optimization using sequential least squares
    def optimizeDual(self, X, Y,):    
        #initialize a variable with size y = m
        a = Variable(len(Y))
        ker = self.kernel_matrix(X)
        
        #update cost parameter 
        Cost = Parameter(sign='positive')
        Cost.value = self.C
    
        #setup constraints
        alphaY = mul_elemwise(Y,a)
        constraints = [a >= 0, a <= Cost, alphaY == 0]
        obj = Maximize(sum_entries(a)-(0.5)*mul_elemwise(Y,a).T * ker * mul_elemwise(Y,a))
        #Add constraints
        print Maximize(sum_entries(a)-(0.5)*mul_elemwise(Y,a).T * ker * mul_elemwise(Y,a)).is_dcp()
           
        return 'not a convex problem'
        prob = Problem(obj,constraints)
        prob.solve(method = 'sdp-relax', solver=MOSEK) 
        print a.value 
        return
       
    def optimizePrimal(self, X, Y):
        m = float(len(Y))
        n = int(np.size(X)/len(Y))
        n = float(n)
        
        w = Variable(np.size(X)/len(Y))
        beta = Variable()
            
        loss = sum_entries(pos(1-mul_elemwise(Y, X*w-beta)))
        reg = (1.0/self.norm)* norm(w, self.norm)
        lambd = Parameter(sign='Positive')
        prob = Problem(Minimize(loss/m + lambd*reg))
        
        lambd.value = 1.0
        prob.solve()
        
        return [w.value, beta.value]
    
    def createArrays(self, train, val, label, split =50):
        train = np.array_split(train, split)[0]

        count_vect = CountVectorizer(stop_words='english', ngram_range=(0,6))
        X_train_counts = count_vect.fit_transform(train['conversation'])
        labels = train['category']
    
        print 'vectorized'
        tf_transformer = TfidfTransformer().fit(X_train_counts)
        X_train_tf = tf_transformer.transform(X_train_counts)
        X_train_tf.shape
        
        X = X_train_tf.toarray()
        Y = (labels.values == label).astype(float)
        
        return [X,Y]
    
    #normalize 
    def normalize(self,col):
        mew = np.mean(col)
        vr = np.var(col)
        sd = np.sqrt(vr)
        if sd == 0:
            return np.ones(np.shape(col))
        return (col-mew)/sd
    
    
    def kernel_linear(self,x1,x2):
        return np.dot(x1,x2.T)
    
    def kernel_quadratic(self,x1,x2):
        return np.dot(x1,x2.T)**2.0
    
    def kernel_gaussian(self,x1,x2,sigma):
        similar = np.exp(np.dot((x1 - x2), (x1 - x2)) / (2.0 * sigma ** 2.0))
        return similar
     
    # calculates gram matrix to be used in dual   
    def kernel_matrix(self, X):
        m = X.shape[0]
        mat = np.zeros((m,m))
        for i in range(m):
            for j in range(m):
                mat[i, j] = self.kernels[self.kernel_type](X[i,:], X[j,:])
        return mat
    
    def kernel_predict(self, x1, X):
        m = X.shape[0]
        #Gram matrix
        mat = np.zeros((1,m))
        for j in range(m):
            mat[j, :] = self.kernels[self.kernel_type](x1, X[j,:])
        return mat
    #make predictions
    def predict(self, X ,w, beta):
        return (np.dot(w.T,X.T) + beta >= 0).astype(float)
        

def main():
    ########################################
    ###Problem predicting all negative class
    ########################################
    train = pickle.load(open('trainset','rb'))
    print len(train)
    val = pickle.load(open('valset','rb'))
    print len(val)

    model = SVM(norm = 2.0, C = 1.0)
    FractionOfData = 100
    split = 2

    labels = ['hockey', 'movies', 'nba', 'news', 'nfl', 'politics', 'soccer', 'worldnews']

    #to determine sizes of X and Y
    [X, Y] = model.createArrays(train, val, 'news', FractionOfData)

    # 1 Vs. all predictions
    #finalPredictions = np.zeros(np.shape(np.array_split(Y,split)))
    for l in labels:
        [X, Y] = model.createArrays(train, val, l, FractionOfData)
        # 50/50 train and test
        X = np.array_split(X, split)
        Y = np.array_split(Y, split)

        X_test = X[1]
        X = X[0]
        Y_test = Y[1]
        Y = Y[0]

        [w, beta] = model.optimizePrimal(X, Y)
        prediction = model.predict(X_test, w, beta)
        print prediction
        #################################
if __name__ == "__main__":
    main()

