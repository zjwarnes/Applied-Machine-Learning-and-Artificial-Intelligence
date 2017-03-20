#the following import allows integer division to become a float
from __future__ import division
import csv
from Cleaning.formatCheck import nameLocation, checkHalfVSFull, halfOrFull, paceInSeconds    
from Cleaning.csvParsing import timeInSeconds
#playing around with some possible features
#for runners over multiple years, tracking multiple features
#since years are not ordered for a particular runner
#most recent features are tracked
fileName = 'Project1_data_no_commas_in_names.csv'
#'lim' controls how many runners to parse through
def parseRunner(lim):
    fileName = "Project1_data_no_commas_in_names.csv"
    with open(fileName, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',', quotechar='|' )
        count = 0
        lastId = 0
        mostRecentYear = 0
        avgPaceTotal = 0
        mostRecentPace = 0
        mostRecentTime = 0
        avgRankTotal = 0
        yearsRun = 1
        maxAge = 0
        gender = ""
        name = ""
        print("ID|NAME|HIGHESTAGE|GENDER|AVGRANK|AVGPACE|MOSTRECENTYEAR|LASTMARATHONTYPE")
        for row in reader:
            #ignore the header of the attribute
            if count == 0:
                count +=1
                continue
            count+=1
            if count > lim:
                break
            #different runner
            if float(row[0]) != lastId:
                if lastId != 0:
                    print(str(lastId)+"|"+name+"|"+str(maxAge)+ \
                          "|"+gender+"|"+str(avgRankTotal/yearsRun)+ \
                          "|"+str(avgPaceTotal/yearsRun)+ \
                          "|"+str(mostRecentYear)+ \
                          "|"+str(halfOrFull(mostRecentPace,mostRecentTime)))
                lastId = float(row[0])
                name = row[1]
                maxAge = float(row[2])
                gender = row[3]
                avgRankTotal = float(row[4])
                avgPaceTotal = paceInSeconds(row[6])
                mostRecentYear = float(row[7])
                mostRecentPace = row[6]
                mostRecentTime = row[5]
                yearsRun = 1
            #same runner, different year
            else:
                if float(row[7]) > mostRecentYear:
                    mostRecentYear = float(row[7])
                    mostRecentPace = row[6]
                    mostRecentTime = row[5]
                if float(row[2]) > maxAge:
                    maxAge = float(row[2])
                yearsRun +=1
                avgPaceTotal += paceInSeconds(row[6])
                avgRankTotal += float(row[4])
            
        csvfile.close()   
        
        
def writeCSV():
    with open('Project1_data_no_commas_in_names.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',', quotechar='|' )
        count = 1
        idx1 = 0
        idx2 = 0
        linesToRemove = checkHalfVSFull()
        namesToRemove = nameLocation()
        earlyBreak = 0
        switch = 0
        with open('Project1_data_formatted.csv', 'wb') as writeFile:
            writer = csv.writer(writeFile)
            for row in reader:
                if idx1 < len(linesToRemove) and idx2 < len(namesToRemove):
                    if linesToRemove[idx1] == count and namesToRemove[idx2] == count:
                        idx1+=1
                        idx2+=1
                        count+=1
                        continue
                if idx1 < len(linesToRemove):
                    if linesToRemove[idx1] == count:
                        idx1+=1
                        count+=1
                        continue
                if idx2 < len(namesToRemove):
                    if namesToRemove[idx2] == count:
                        idx2+=1
                        count+=1
                        continue
                #added for linear regression    
                
                if switch == 1:
                    row.append(paceRatio(row[0], row[7]))    
                switch = 1
                
                writer.writerow(row)
                count+=1
                
                earlyBreak+=1
                if earlyBreak == 40000:
                    break
    csvfile.close()
    writeFile.close()
 
 
def searchYears(years,i,firstYear):
    for y in years:
        if float(y) == (float(firstYear) + i):
            return True
    return False
     
def yearRatio(ID, year):
    with open('Project1_data_formatted.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',', quotechar='|' )
        years = []
        switch = 0
        for row in reader:
            if row[0] != ID:
                if switch == 1:
                    break
                continue
            switch = 1
            #collect each of the years for a particular runner
            years.append(float(row[7]))
        csvfile.close()
        
        num = 0
        firstYear = year
        for y in years:
            #how many years attended before this year
            if y < float(year):
                if y < firstYear:
                    firstYear = y
                num +=1
        if float(year) == float(firstYear):
            return 0
        else:
            return num/(float(year)-float(firstYear))

def paceRatio(ID, year):
    with open('Project1_data_no_commas_in_names.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',', quotechar='|' )
        times = []
        years = []
        lastTime = 0
        lastYear = 0
        #to stop reading after runner is found
        switch = 0
        for row in reader:
            if row[0] != ID:
                if switch == 1:
                    break
                continue
            switch = 1
            
            #collect each of the years for a particular runner
            #collect times in hours
            if year > row[7] and lastYear < float(row[7]):
                lastYear = row[7]
                lastTime = timeInSeconds(row[5])/3600
            times.append(timeInSeconds(row[5])/3600)
            years.append(float(row[7]))
        csvfile.close()
        
        num = 0
        totTime = 0
        for i in range(len(years)):
            #average of times before this year
            if years[i] < float(year):
                totTime+=times[i]
                num +=1
        if num == 0:
            return 1
        else:
            #compare last run to average of all runs
            return lastTime/(totTime/num)
 
# the goal for this function is to take the formatted file 
#and create the negative examples for logistic regression
#features include, age, gender, and years participated
def classificationFeatures():
    with open('Project1_data_formatted.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',', quotechar='|' )
        with open('Project1_data_logisticFeatures.csv', 'wb') as writeFile:
            writer = csv.writer(writeFile)
            
            earlyBreak = 0
            firstLine = 0
            
            years = []
            
            currId = 0
            currName = ''
            currGender = ''
            firstYear = 0
            firstAge = 0
            
            for row in reader:
                #If I'm still reading the same racer id
                #positive examples
                if row[0] == currId:
                    years.append(row[7])
                    #store the last year run to calculate age
                    if float(row[7]) < firstYear:
                        firstYear = float(row[7]) 
                        firstAge = float(row[2])
                    
                #new Id to check
                else:
                    #adding the new entries of last runner
                    # != 13 in case someone has run all 13 years
                    if len(years) != 13:
                        if firstLine == 0:
                            firstLine +=1
                            continue
                        if firstLine == 1:
                            firstLine +=1
                            #store the info of the first runner
                            currId = row[0]
                            currName = row[1]
                            firstAge = row[2]
                            currGender = row[3]
                            firstYear = row[7]
                    
                            #add the year to the list
                            years = [row[7]]
                            row.append(1)
                            row.append(yearRatio(row[0], row[7]))
                            writer.writerow(row)  
                            continue;
                        
                        #used to calculate attendance ratio
                        count = 0
                        #if a runner is not present for more than 3 years in a row
                        #stop recording
                        absent = 0
                        #only write missing entries for someone
                        for i in range(0,2017-float(firstYear)):
                            newRow = []
                            #if the entry for that year already exists
                            if searchYears(years, i,firstYear):
                                count+=1
                                #reset the absent clock
                                absent=0
                                continue
                            absent+=1
                            if absent == 3:
                                break
                            
                            #create a new row, entry for that year does not exist
                            newRow.append(currId)
                            newRow.append(currName)
                            #this gives me a value to add to first age
                            newRow.append(float(firstAge)+i)
                            newRow.append(currGender)
                            newRow.append(0)
                            newRow.append('00:00:00')
                            newRow.append('00:00')
                            newRow.append(float(firstYear)+i)
                            newRow.append(0)           #did not attend
                            #ratio of years attended to current year
                            #i+1 since the first year should be included in the ratio
                            if i != 0:
                                newRow.append(count/(i))
                            else:
                                newRow.append(0)
                            writer.writerow(newRow)
                    
                    #store the info of the next runner
                    currId = row[0]
                    currName = row[1]
                    firstAge = row[2]
                    currGender = row[3]
                    firstYear = row[7]
                    
                    #add the year to the list, re-initialize it
                    years = [row[7]]
                    
                row.append(1)
                row.append(yearRatio(currId,row[7]))
                writer.writerow(row)  
                
                earlyBreak +=1
                if earlyBreak == 40000:
                    break
                 
    csvfile.close()
    writeFile.close()
     
#writeCSV()    
#classificationFeatures()