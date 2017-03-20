import csv
import numpy as np

class runner:
    def __init__(self, Id, name, ages, gender, times, years):
        self.Id = Id
        self.name = name
        self.ages = ages
        self.gender = gender
        self.times = times
        self.years = years
      
def createRunnerDatabase(filename, sze):
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',', quotechar='|' )
        
        #helpful variables
        dictionary = {}
        count = 0
        currentID = 0
        
        #initialize some variables
        ID = 0
        name = ''
        ages = []
        gender = ''
        times = []
        years = []
        for row in reader:
            #skip first line
            if count == 0:
                count+=1
                continue
            
            #same runner
            if row[0] == currentID:
                currentID = row[0]
                ages.append(row[2])
                times.append(row[5])
                years.append(row[7])
                
            else:  
                #first runner
                if count == 1:
                    currentID = row[0]
                    ID = currentID
                    name = row[1]
                    ages = [row[2]]
                    gender = row[3]  
                    times = [row[5]]
                    years = [row[7]]
                
                #new runner
                else:
                    #insert last runner's data
                    r = runner(id,name,ages,gender,times,years)
                    dictionary.update({ID : r})
                    
                    currentID = row[0]
                    ID = currentID
                    name = row[1]
                    ages = [row[2]]
                    gender = row[3]  
                    times = [row[5]]
                    years = [row[7]]
            
            count +=1 
            #break early
            if count > sze:
                break        
        #insert the last runner
        r = runner(id,name,ages,gender,times,years)
        dictionary.update({ID : r})    
          
        csvfile.close()
        return dictionary

def calcAttendanceRatio(dictionary, ID, year):
    years = dictionary[str(ID)].years
    firstYear = min(years)
    times = dictionary[str(ID)].times
    
    count = 0.0
    totYears = 0.0
    for y in range(int(firstYear), int(year)):
        if y < 2013:
            continue
        #don't count the negative entries
        #if timeInSeconds(times[indexOf(years, str(y))]) >10:
        #    continue 
        if str(y) in years:
            count+=1
        totYears+=1
    
    #avoiding division by 0
    if totYears == 0.0:
        return 0.0
    return count/totYears

def timeInSeconds(entry):
    HMS = entry.split(":")
    seconds = float(HMS[2])
    seconds += float(HMS[1])*60
    seconds += float(HMS[0])*3600
    return seconds

def calcImprovementRatio(dictionary, ID, year):
    yrs = dictionary[str(ID)].years
    tme = dictionary[str(ID)].times
    if len(yrs) == 1:
        return 1.0
    #remove time and year of desired year
    idx = indexOf(yrs, str(year))
    years = list(yrs)
    times = list(tme)
    
    years.remove(yrs[idx])
    times.remove(tme[idx])
    
    count = 0.0
    totTime = 0
    mostRecentYear = 0  
    for y in years:
        #don't count the negative entries
        if timeInSeconds(times[indexOf(years, str(y))]) <10:
            continue 
        #add together previous years
        if int(y) < int(year):
            #track most recent year
            if int(y) > mostRecentYear:
                mostRecentYear = y
            #add the corresponding time
            idx = indexOf(years,str(y))
            totTime+= timeInSeconds(times[idx])
            count+=1        
    #find most recent time
    idx = indexOf(years,str(mostRecentYear))
    mostRecentTime = timeInSeconds(times[idx])
    
    if mostRecentYear == 0:
        return 1.0
    averageTime = float(totTime/count)
    return mostRecentTime/averageTime
    
def indexOf(lst, st):
    for idx, val in enumerate(lst):
        if val == st:
            return idx
    return -1

def sex(st):
    if st == 'M':
        return 1.0
    elif st == 'F':
        return 0.5
    else:
        return 0.0
 
def addNegatives(dictionary,sze):
    for i in range(1,sze):
        #only place negative values every few entries
        if i % 3 == 0:
            continue
        #missing ids
        if dictionary.get(str(i)) == None:
            continue
        years = dictionary[str(i)].years
        ages = dictionary[str(i)].ages
    
        #should be the same, assumed to be continuous
        maxAge = float(max(ages))
        maxYear = float(max(years))
            
        ###Add values to the dictionary
        ##add at most three years
        for q in range(0,1):
            year = 2016 - q
            #year already exists
            if str(year) in years:
                continue
            #calculate age
            #i.e. if max age is 40 and max year is 2015, curAge = 40 - (2015 - 2016)
            dif = maxYear - year
            curAge = maxAge - dif
                
            #add new entry
            dictionary[str(i)].times.append('0:00:00')
            dictionary[str(i)].years.append(str(year))
            dictionary[str(i)].ages.append(str(curAge))
        
#id,name,age,gender, times, years 
#feature 1 = age, 2 = gender, 3 = times, 4 = years, 5 = improvementRatio, 6 = attendanceRatio
def createXArray(dictionary, sze, features, negSwitch):
    di = {1:'age', 2:'gender', 3:'times', 4:'years', 5:'improvementRatio', 6:'attendanceRatio', 7:'attended'}
    mtrix = np.empty([1,len(features)])
    #only add negatives once
    if negSwitch == 1:
        addNegatives(dictionary, sze)
    for i in range(1,sze):
        if dictionary.get(str(i)) == None:
            continue
        
        
        
        #retrieve the new lists, after new entries added
        ages = dictionary[str(i)].ages   
        gender = dictionary[str(i)].gender
        times = dictionary[str(i)].times   
        years = dictionary[str(i)].years
        attendanceRatio = []
        for k in range(len(times)):
            attendanceRatio.append(calcAttendanceRatio(dictionary, i, years[k]))
        
        improvementRatio = []
        for l in range(len(years)):
            improvementRatio.append(calcImprovementRatio(dictionary, i, years[l]))
        
        #build the matrix of chosen features
        for j in range(len(ages)):
            n = np.empty([1,1])
            for p in range(len(features)):
                if di[features[p]] == 'age':
                    n = np.append(n,float(ages[j]))
                    
                elif di[features[p]] == 'gender':
                    n = np.append(n,float(sex(gender)))
                    
                elif di[features[p]] == 'times':
                    n = np.append(n,float(timeInSeconds(times[j])))
                    
                elif di[features[p]] == 'years':
                    n = np.append(n,float(years[j]))
                    
                elif di[features[p]] == 'improvementRatio':
                    n = np.append(n,float(improvementRatio[j]))
                    
                elif di[features[p]] == 'attendanceRatio':
                    n = np.append(n,float(attendanceRatio[j]))
                    
            m = np.array([n[1:np.size(n)]])
            mtrix = np.append(mtrix, m, axis = 0)
       
    #remove empty row (values are random in this row initially) 
    mtrix = mtrix[1:np.size(mtrix), :]
    
    #add bias terms to each row
    mtrix = np.append(np.transpose(np.array([np.ones(np.size(mtrix[:,0]))])), mtrix, axis=1)
    return mtrix
    #return mtrix[1:np.size(mtrix), :]
    
#1 = attendance, 2 = times
def createYArray(dictionary, sze, typ):
    mtrix = np.empty([1,1])
    if typ == 1:
        for i in range(1,sze):
            if dictionary.get(str(i)) == None:
                continue
            #1 if time != 0, 0 if the time == 0
            times = dictionary[str(i)].times
            for j in range(len(times)):
                if timeInSeconds(times[j]) > 0:
                    m = np.array([[1]])
                else:
                    m = np.array([[0]])  
                mtrix = np.append(mtrix, m, axis = 0)   
    elif typ == 2:
        for i in range(1,sze):
            if dictionary.get(str(i)) == None:
                continue
            times = dictionary[str(i)].times
            for j in range(len(times)):
                m = np.array([[float(timeInSeconds(times[j]))]])
                mtrix = np.append(mtrix, m, axis = 0)
    return mtrix[1:np.size(mtrix), :]

def createPredictionXArray(dct,sze,features):
    di = {1:'age', 2:'gender', 3:'times', 4:'years', 5:'improvementRatio', 6:'attendanceRatio'}
    mtrix = np.empty([1,len(features)])
    for i in range(1,sze):
        if dct.get(str(i)) == None:
            continue
        maxYear = float(max(dct[str(i)].years))
        maxAge = float(max(dct[str(i)].ages))
        
        age = maxAge+(2017-maxYear)
        gender = dct[str(i)].gender
        improvement = calcImprovementRatio(dct, i, 2017)
        attendance = calcAttendanceRatio(dct, i, 2017)
        
        n = np.empty([1,1])
        for p in range(len(features)):
            if di[features[p]] == 'age':
                n = np.append(n,float(age))
            elif di[features[p]] == 'gender':
                n = np.append(n,float(sex(gender)))
            elif di[features[p]] == 'years':
                    n = np.append(n,2017)
            elif di[features[p]] == 'improvementRatio':
                n = np.append(n,float(improvement))
            elif di[features[p]] == 'attendanceRatio':
                n = np.append(n,float(attendance))
                
        m = np.array([n[1:np.size(n)]])
        mtrix = np.append(mtrix, m, axis = 0)
    mtrix = mtrix[1:np.size(mtrix), :]
    mtrix = np.append(np.transpose(np.array([np.ones(np.size(mtrix[:,0]))])), mtrix, axis=1)    
    return mtrix

