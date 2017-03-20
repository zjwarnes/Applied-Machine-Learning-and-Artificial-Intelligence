import csv
import sys
import random
import math
import copy


# use only ID's, no point in names since every name maps to a unique ID.
#try catch needed?
def loadCsv(filename):
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	return dataset

#split data into a training set and a validation set
def splitDataset(dataset):
	size = int(len(dataset) * 0.20)
	tSet = []
	vSet = list(dataset)
	while len(tSet) < size:
		i = random.randrange(len(vSet))
		tSet.append(vSet.pop(i))
	return [tSet, vSet]

#make dictionary split by year
def separateByYears(dataset, columnID):
	separated = {}
	years = {}
	for i in range(1, len(dataset)):
		line = dataset[i]
		if (line[columnID] not in separated):
			separated[line[columnID]] = []
			years[line[columnID]] = []
		separated[line[columnID]].append(line)
	return separated, years

#mean
def mean(set):
	return (sum(set)/float(len(set)))

#stddev
def stdev(set):
	if (len(set)-1) == 0:
		return float(0.0)
	average = mean(set)
	variance = (sum([(pow(y - average, 2)) for y in set]) / float(len(set)-1))
	return math.sqrt(variance)



#### Predictions ####

#probability density function, probability of x being in the normal with mean and stdev
def calculateProbability(x, mean, stdev):
	if stdev == 0.0:
		stdev = 1.0
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent



def paceColumn(dataset):
	l = []
	for i in range(1, len(dataset)):
		l.append(float(dataset[i][6]))
	return l
'''
def getAgeRange(line):
	if line in range(0,10):
		return '0'
	elif line in range(10,20):
		return '10'
	elif line in range(20,30):
		return '20'
	elif line in range(30,40):
		return '30'
	elif line in range(40,50):
		return '40'
	elif line in range(50,60):
		return '50'
	elif line in range(60,70):
		return '60'
	elif line in range(70,80):
		return '70'
	elif line in range(80,90):
		return '80'
	elif line in range(90,100):
		return '90'
	return '-1'
def ageColumnYears(yearData, years):
	temp = {}
	probabilities = [float(0.0)]*10
	for i in range(10):
			temp[str(i*10)] = []
	totalcount = 0.0
	sum1 = float(0.0)
	for year in yearData:
		l = []
		for i in range(len(yearData[year])):
			l.append(float(yearData[year][i][2]))
			temp[getAgeRange(float(yearData[year][i][2]))].append([yearData[year][i]])
			totalcount = totalcount + 1.0		
		#temp1 = mean(l)
		#temp2 = stdev(l)
		#years[year].append(temp1)
		#years[year].append(temp2)
	for i in range(10):
			probabilities[i] = len(temp[str(i*10)])/(totalcount)
	print probabilities
	print sum1
	return years
'''


def ageGivenPace(ageData, pace):
	probabilities = {}
	bestProb = 0.0
	bestAge = None
	for age in ageData:
		probabilities[age] = 1
	for age in ageData:
		probabilities[age] *= calculateProbability(pace, ageData[age][0], ageData[age][1])
	for age in probabilities:
		if probabilities[age] != 1 and probabilities[age] > bestProb:
			bestProb = probabilities[age]
			bestAge = age
	return bestAge

def nextRaceGivenHistory(trainingData, instance, IDsList):
	probabilities = {}
	bestProb = 0.0
	bestResult = None
	for binary in trainingData:
		probabilities[binary] = 1
	for binary in trainingData:
		probabilities[binary] *= calculateProbability(float(instance[6]), trainingData[binary][4], trainingData[binary][5])
		probabilities[binary] *= calculateProbability(float(instance[2]), trainingData[binary][0], trainingData[binary][1])
		probabilities[binary] *= calculateProbability(float(instance[4]), trainingData[binary][2], trainingData[binary][3])
		probabilities[binary] *= calculateProbability(IDsList[instance[3]][-1], trainingData[binary][6], trainingData[binary][7])
	for binary in probabilities:
		if probabilities[binary] != 1 and probabilities[binary] > bestProb:
			bestProb = probabilities[binary]
			bestResult = binary
	return bestResult

def columnYears(data, labels, columnID):
	for ID in columnID:
		for line in data:
			l = []
			for i in range(len(data[line])):
				l.append(float(data[line][i][ID]))
			temp1 = mean(l)
			temp2 = stdev(l)
			labels[line].append(temp1)
			labels[line].append(temp2)
	return labels

def getNextYear(data, year, ID):
	for i in range(len(data)):
		if int(ID) == int(data[str(int(year)+1)][i][3]):
			return 1
	return 0

def checkAccuracy(meansAndDevs, Instances, IDData):
	correct = 0.0000001
	incorrect = 0.0000001
	count = 1
	for Instance in Instances:
		if count == 1:
			count = 2
			continue
		temp = nextRaceGivenHistory(meansAndDevs, Instance, IDData)
		if temp == Instance[-1]:
			correct = correct + 1
		else:
			incorrect = incorrect + 1
	print "Accuracy: " + str(correct / (correct + incorrect)*100)+"%"

def predictionSet(meansAndDevs, IDData, IDs):
	for instance in IDData:
		temp = getRecentYear(instance, IDData)
		IDs[instance] = predict(meansAndDevs, temp, IDData)
	return IDs

def predict(meansAndDevs, temp, IDData):
	return nextRaceGivenHistory(meansAndDevs['next'], temp, IDData)

def getRecentYear(IDSet, IDData):
	newestYear = 2003
	instance = []
	for year in range(0,(len(IDData[IDSet])-1)):
		if IDData[IDSet][year][7] >= newestYear:
			newestYear = IDData[IDSet][year][7]
			instance = IDData[IDSet][year]
	return instance

# ++ and #value in dataset but yolo
def main():
	dataset = loadCsv("aaadata.csv")
	yearData, years = separateByYears(dataset, 7)
	#print yearData['2006']

	#get NextYear or not
	count = 0
	for j in range(0,13):
		if j == 9 or j == 10:
				j = j+2
		for i in range(0, len(yearData[str(2003+j)])):
			if getNextYear(yearData, 2003+j, yearData[str(2003+j)][i][3]) == 1:
				for a in range (1, len(dataset)):
					if int(dataset[a][7]) == 2003+j and int(dataset[a][3]) == int(yearData[str(2003+j)][i][3]):
						count = count + 1
						dataset[a][8] = '1'
	print count

	ageData, ages = separateByYears(dataset, 2)
	nextRaceData, nextRace = separateByYears(dataset, 8)
	IDData, IDs = separateByYears(dataset, 3)
	listOfCount = []
	listOfNoCount = []
	for ID in IDData:
		num = len(IDData[ID])
		IDData[ID].append(num)
	for ID in IDData:
		for i in range(0,(len(IDData[ID])-1)):
			if IDData[ID][i][8] == '1' :
				listOfCount.append(IDData[ID][-1])
			if IDData[ID][i][8] == '-1':
				listOfNoCount.append(IDData[ID][-1])
	print IDData['5'][-1]
	meansAndDevs = {}
	meansAndDevs['pace'] = []
	meansAndDevs['age'] = []
	meansAndDevs['rank'] = []
	meansAndDevs['next'] = []
	for i in meansAndDevs:
		meansAndDevs[i] = copy.deepcopy(years)

	#meansAndDevs['pace'] = columnYears(yearData, copy.deepcopy(years), 6)
	#meansAndDevs['age'] = columnYears(ageData, copy.deepcopy(ages), 6)
	meansAndDevs['next'] = columnYears(nextRaceData, copy.deepcopy(nextRace), [2,4,6])
	meansAndDevs['next']['1'].append(mean(listOfCount))
	meansAndDevs['next']['1'].append(stdev(listOfCount))
	meansAndDevs['next']['-1'].append(mean(listOfNoCount))
	meansAndDevs['next']['-1'].append(stdev(listOfNoCount))
	checkAccuracy(meansAndDevs['next'], dataset, IDData)
	#nextRaceGivenHistory(meansAndDevs['next'], IDData['4'][0], IDData)
	#temp = ageGivenPace(meansAndDevs['age'], 590.0)
	#meansAndDevs['rank'] = columnYears(yearData, copy.deepcopy(years), 4)
	#ageColumnYears(yearData, years)
	#tSet, valSet = splitDataset(dataset)
	store = predictionSet(meansAndDevs, IDData, IDs)
	with open('output.csv', 'wb') as csvfile:
		writer = csv.writer(csvfile, delimiter=',' )
		for key, value in store.items():
			writer.writerow([key, value])
	csvfile.close()
if __name__ == "__main__":
    main()


