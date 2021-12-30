from datetime import date, datetime, timedelta
from itertools import count
import numpy as np
import os
import pandas as pd
import psutil

# countPerDay = 85 # one quote data every 5 minutes, total it is 85 quotes per day per symbol
countPerDay = 29 # as it is resampled every 15 minutes, the countPerDay was changed from 85 to 29
symbolCount = 123
signalCount = 12 
defaultDatePattern = '%Y%m%d'
quotesOnedayPath = "data/quotes_oneday_data"
targetSymbolIndex = 99 # target symbol is TECL, its index is 99 (starting from 0) in the dataframe for the same time
minuteFilterArray = [0, 15, 30, 45] # every 15 minutes
# minuteFilterArray = [0, 10, 20, 30, 40, 50] # every 10 minutes

def getMemoryUsed():
    return int(psutil.virtual_memory()[3] / 1024 /1024)

def getFile(day):
    filename = 'data_' + datetime.strftime(day.to_timestamp(), defaultDatePattern) + ".csv"
    return os.path.join(quotesOnedayPath, filename)

def minuteFilter(row):
    currentTime = datetime.strptime(row['currentTime'], "%Y-%m-%d %H:%M:%S")
    return currentTime.minute in minuteFilterArray

"""
- dropColumns shall be True, False is for testing purpose
"""
def readToDataFrame(startDate, endDate):
    periodRange = pd.period_range(start=startDate, end=endDate, freq='D')
    result = []
    for day in periodRange:
        file = getFile(day)
        if os.path.isfile(file):
            print(f'read file for {day}')
            df = pd.read_csv(file)
            df = df[df.apply(minuteFilter, axis=1)]
            result.append(df)
    return pd.concat(result)

# tick starts from 0 
def getArray(df, tick):
    subDataFrame = getSubDataFrame(df, tick)

    # convert symbol to int or drop symbol?
    # convert date to int
    subDataFrame = subDataFrame.drop(columns=['symbol', 'currentTime'])
    
    # convert to np array
    data = []
    for i in range(countPerDay):
        startIndex = i * symbolCount
        endIndex = (i+1) * symbolCount 
        if startIndex >= len(subDataFrame):
            return None
        data.append(subDataFrame[startIndex : endIndex].to_numpy().tolist())
    return data

"""
tick starts from 0
"""
def getSubDataFrame(df, tick):
    startIndex = tick * symbolCount
    endIndex = startIndex + countPerDay * symbolCount
    return df[startIndex:endIndex]

def printSample(data):
    print('data shape: ', np.array(data).shape)
    print(data[0][0][0], data[0][0][1])
    print(data[0][122][0], data[0][122][1])
    print(data[countPerDay-1][0][0], data[countPerDay-1][0][1])
    print(data[countPerDay-1][122][0], data[countPerDay-1][122][1])

if __name__ == '__main__':
    startDate, endDate = '20210831', '20210901'
    memoryBefore = getMemoryUsed()
    
    df = readToDataFrame(startDate, endDate)
    print(len(df))
    
    memoryAfter = getMemoryUsed()
    print(f'memory used {memoryAfter - memoryBefore}M')

    data = getArray(df, 0)
    printSample(data)
    # assert data[0][99][0] == 'TECL'
    
    data = getArray(df, 1)
    printSample(data)
    # assert data[0][99][0] == 'TECL'

    print(type(data))
    print(type(data[0]))
    print(type(data[0][0]))
    print(type(data[0][0][0]))
    
    assert getArray(df, 100) is None
