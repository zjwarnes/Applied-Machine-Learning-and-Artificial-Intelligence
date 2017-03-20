import csv
import numpy as np
#converting the time format into a numerical value
def timeInSeconds(entry):
    HMS = entry.split(":")
    seconds = float(HMS[2])
    seconds += float(HMS[1])*60
    seconds += float(HMS[0])*3600
    return seconds
#Returns an array of the desired attribute field, ignoring the header
#parses through a csv file, for a particular number of rows 'numEntries'
def parseAttribute(index, fileName, numEntries):
    with open(fileName, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',', quotechar='|' )
        count = 0;
        data = []
        for row in reader:
            #ignore the header of the attribute
            if count == 0:
                count +=1
                continue
            count+=1
            if count > numEntries:
                break
            data.append(row[index])
            #print(row[index])
            #print(', '.join(row))
        return data
        csvfile.close()
        


def createArray(sze, fileName, features):
    #age, year any other feature you want as an input
    #currently only have working for age and year
    Xinput = features
    #initialize size of feature space, with bias term
    # to be inverted at the end
    
    switch = 0 
    for i in Xinput:
        if switch == 0:
            switch =+1
            lgth = parseAttribute(0, fileName, sze+1)
            mtrix = np.array([np.ones(len(lgth))])
        #condition for number values
        if i in [2,4,7,8,9]:
            arr = parseAttribute(i, fileName, sze+1)
            for j in range(len(arr)):
                if i in [8,9]:
                    arr[j] = float(arr[j])
                else:
                    arr[j] = float(arr[j])
            vec = np.array([np.ones(len(arr))])
            for k in range(len(arr)):
                vec[0][k] = arr[k]
            mtrix = np.append(mtrix, vec, axis = 0)
             
        if i == 3:
            arr = parseAttribute(i, fileName, sze+1)
            for j in range(len(arr)):
                if arr[j] == 'M':
                    arr[j] = 1
                elif arr[j] == 'F':
                    arr[j] = 0
                else:
                    arr[j] = 0.5
            vec = np.array([np.ones(len(arr))])
            for k in range(len(arr)):
                vec[0][k] = arr[k]   
            
            mtrix = np.append(mtrix, vec, axis = 0)
        
    return np.transpose(mtrix)    
    
def createOutputs(sze, fileName, typ):
    #create vector of size n to be updated with race times in seconds
    #return output variable for logistic regression
    if typ == 1:
        att = parseAttribute(8, fileName, sze+1)
        for j in range(len(att)):
            att[j] = float(att[j])
        Vector = np.array([np.ones(len(att))])
        for k in range(len(att)):
            Vector[0][k] = att[k]   
        return np.transpose(Vector)  
    
    #return output variable for linear regression
    if typ == 2:
        tme = parseAttribute(5, fileName, sze+1)
        for j in range(len(tme)):
            tme[j] = float(timeInSeconds(tme[j]))
        Vector = np.array([np.ones(len(tme))])
        for k in range(len(tme)):
            Vector[0][k] = tme[k]   
        return np.transpose(Vector)  
    

fileName = 'Project1_data_formatted.csv'
#print createArray(100,  fileName)
#print createOutputs(100, fileName)