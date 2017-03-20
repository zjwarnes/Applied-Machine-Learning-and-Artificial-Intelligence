from Cleaning.csvParsing import parseAttribute
'''
id = numbers 1 to max number of runners in data set 

name = characters all lower case

age = check for numbers outside a typical running range say <10, >100

sex = either M or F

rank = no more than the total runners can't be negative

time = no more than the fastest times for M and F and nothing to high
~nothing over 10 hours

pace = same as time

year = nothing over 15 years ago
'''

import re
#fileName = 'Project1_data.csv'
fileName = 'Project1_data_no_commas_in_names.csv'


#converting the time format into a numerical value
def timeInSeconds(entry):
    HMS = entry.split(":")
    seconds = int(HMS[2])
    seconds += int(HMS[1])*60
    seconds += int(HMS[0])*3600
    return seconds

#similar to timeInSeconds
def paceInSeconds(entry):
    MS = entry.split(":")
    seconds = int(MS[1])
    seconds += int(MS[0])*60
    return seconds


#all function have the same format, if a pattern doesn't match what I expect
#or an earlier entry affects a later one then it prints out that entry
def checkID():
    data = parseAttribute(0,fileName,100000)
    for entry in data:
        if int(entry) > 30417 or int(entry) < 0:
            print(entry)

def checkName():
    rgx = '^([a-z A-Z\-\.\'\"]+)$'
    #uncomment below to all names with non-alphabet chars
    #rgx = '^([a-z A-Z]+)$'
    data = parseAttribute(1,fileName,100000)
    line = 2
    for entry in data:
        m = re.search(rgx,entry)
        if m == None:
            print('Line: ' + str(line) + '  ' + entry)
        line +=1
        
def checkAge():
    #checks for any age >=100 and <=10
    #oldest runners were 98, youngest were 10
    rgx = '^([1-9]?[0-9])$'
    data = parseAttribute(2,fileName,100000)
    line = 2
    for entry in data:
        m = re.search(rgx,entry)
        if m == None:
            print('Line:' + str(line) + '   ' +entry)
        line +=1
        
def checkSex():
    #about 20 or so runners have gender undefined 'U'
    rgx = '^[MF]$'
    data = parseAttribute(3,fileName,100000)
    for entry in data:
        m = re.search(rgx,entry)
        if m == None:
            print(entry)

def checkRank():
    #no one is ranked over 4000
    rgx = '^[0-3]?[0-9]?[0-9]?[0-9]?$'
    data = parseAttribute(4,fileName,100000)
    line = 2
    for entry in data:
        m = re.search(rgx,entry)
        if m == None:
            print('Line:'+str(line)+ '   ' + entry)
        line+=1    
        
def checkTime():
    #world record is just over 2 hours
    #times under two hours may be for half marathon
    rgx = '^(0?[2-9]:\d{2}:\d{2})$'
    data = parseAttribute(5,fileName,100000)
    line = 2
    cheaters = 0
    for entry in data:
        m = re.search(rgx,entry)
        if m == None:
            print('Line:'+str(line)+ '   ' + entry)
            cheaters += 1
        line+=1 
    print(cheaters)

def checkPace():
    #using the world record, the pace is 4:36 miles
    #anything below this should not exist, also anything too high
    rgx = '^(((1[0-9])|[4-9]):[0-5]\d)$'
    data = parseAttribute(6,fileName,100000)
    line = 2
    for entry in data:
        m = re.search(rgx,entry)
        if m == None:
            print('Line:'+str(line)+ '   ' + entry)
        line +=1
    
def checkYear():
    #data should only be for the last 15 years 'as stated in project'
    rgx = '^(20(1[0-6]|0[4-9]))$'
    data = parseAttribute(7,fileName,100000)
    line = 2
    for entry in data:
        m = re.search(rgx,entry)
        if m == None:
            print('Line:'+str(line)+ '   ' + entry)
        line +=1  
        
#pace times are per mile, so multiplying by a constant
#to determine if the marathon was a half or a full   
#can be easily modified to determine the entries to remove     
def checkHalfVSFullOriginal():
    entries = 40000
    dataPace = parseAttribute(6, fileName, entries)
    dataTime = parseAttribute(5, fileName, entries)
    idx = 0
    line = 2
    fullRun = 0
    halfRun = 0
    count = 2
    lines = []
    for p in dataPace:
        time = timeInSeconds(dataTime[idx])
        
        full = paceInSeconds(p)* 26.219
        half = full/2
        
        if abs(full-time) < abs(half-time):
            fullRun+=1
        else:
            halfRun+=1
            lines.append(count)
        line+=1
        idx+=1
        count+=1
    #number of each type of runner
    #print("Full marathons:" +str(fullRun))
    #print("Half marathons:" + str(halfRun))
    return lines

def checkHalfVSFull():
    entries = 40000
    dataPace = parseAttribute(6, fileName, entries)
    dataTime = parseAttribute(5, fileName, entries)
    idx = 0
    line = 2
    count = 2
    lines = []
    for p in dataPace:
        time = timeInSeconds(dataTime[idx])
        full = paceInSeconds(p)* 26.219
        half = full/2
        if abs(full-time) >= abs(half-time):
            lines.append(count)
        line+=1
        idx+=1
        count+=1
    return lines


def nameLocation():
    rgx = '^(private)$'
    data = parseAttribute(1,fileName,100000)
    line = 2
    idx = 0
    lines = []
    for entry in data:
        m = re.search(rgx,entry)
        if m != None:
            lines.append(line)
            idx+=1
        line +=1
    return lines


#Used to relate pace to time 
#in order to determine half vs full marathon
def halfOrFull(pace,time):
    t = timeInSeconds(time)
    full = paceInSeconds(pace)* 26.219
    half = full/2
        
    if abs(full-t) < abs(half-t):
        return "Full"
    else:
        return "Half"



