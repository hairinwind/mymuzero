from datetime import date, datetime, timedelta
import numpy as np
import os
import pandas as pd

defaultDatePattern = '%Y%m%d'
quotesOnedayPath = "data/quotes_oneday_data"

## input argument day: 20210831
## returns 20210901
def addOneDay(day):
    day1 = datetime.strptime(day, defaultDatePattern)
    nextDay = day1 + timedelta(days=1)
    return datetime.strftime(nextDay, defaultDatePattern)

class LearningDataReader: 

    def __init__(self, startDate, endDate):
        self.startDate = startDate
        self.endDate = endDate
        self.data = None
        self.comingData = []

    def __init__(self, config):
        print(f'init LearningDataReader with config {config}')
        self.startDate = config['startDate']
        self.endDate = config['endDate']
        self.data = None
        self.comingData = []

    def readLearningData(self, date):
        print(f'read learning data of {date}')
        self.currentReadingDay = date
        df = pd.read_csv(os.path.join(quotesOnedayPath, self.getFileName(date)))
        groupByTime = df.groupby('currentTime')
        data = []
        for group in groupByTime:
            # group is a tuple, key is the groupby column value, value is the groupby dataframe
            # group[1].sorted
            onetimeData = group[1]
            onetimeData.drop(columns='symbol')
            data.append(onetimeData[onetimeData.columns[2:]].to_numpy())
        return data

    def getFileName(self, date):
        return 'data_' + date + '.csv'

    def fileExists(self, date):
        return os.path.isfile(os.path.join(quotesOnedayPath, self.getFileName(date)))

    def nextReadingDay(self, day): 
        nextDay = addOneDay(day)
        while not self.fileExists(nextDay):
            if nextDay > self.endDate:
                return
            nextDay = addOneDay(nextDay)
            # print(f'nextDay {nextDay}')
        return nextDay            

    def next(self):
        if self.data is None:
            self.data = self.readLearningData(self.startDate)
            return self.data
        if not self.comingData:
            self.currentReadingDay = self.nextReadingDay(self.currentReadingDay) 
            if not self.currentReadingDay:
                return None
            self.comingData = self.readLearningData(self.currentReadingDay)
        self.data = np.roll(self.data, -1, axis=0) # roll the first element to the last element
        self.data[-1] = self.comingData.pop(0)
        return self.data

if __name__ == '__main__':
    startDate = '20210831'
    endDate = '20211130'
    leanringDataReader = LearningDataReader(startDate, endDate)
    
    count = 0
    data1 = None
    data2 = None
    while True:
        data = leanringDataReader.next()
        if data is None:
            break;
        count = count + 1
        if count == 2:
            data1 = data
            print(np.array(data1).shape)
        if count == 3:
            data2 = data
            print(f'data1 {data1[1]}')
            print(f'data2 {data2[0]}')
            assert np.array_equal(data1[1], data2[0])
    
    print(f'total record {count}')
        
