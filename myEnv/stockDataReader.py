from datetime import date, datetime, timedelta
from itertools import count
import numpy as np
import os
import pandas as pd
import psutil

countPerDay = 85 # one quote data every 5 minutes, total it is 85 quotes per day per symbol
symbolCount = 123 
defaultDatePattern = '%Y%m%d'
quotesOnedayPath = "data/quotes_oneday_data"

def getMemoryUsed():
    return int(psutil.virtual_memory()[3] / 1024 /1024)

def getFile(day):
    filename = 'data_' + datetime.strftime(day.to_timestamp(), defaultDatePattern) + ".csv"
    return os.path.join(quotesOnedayPath, filename)

"""
- dropColumns shall be True, False is for testing purpose
"""
def readToDataFrame(startDate, endDate, dropColumns=True):
    periodRange = pd.period_range(start=startDate, end=endDate, freq='D')
    result = []
    for day in periodRange:
        file = getFile(day)
        if os.path.isfile(file):
            print(f'read file for {day}')
            df = pd.read_csv(file)
            # convert symbol to int or drop symbol?
            # convert date to int
            if dropColumns:
                df = df.drop(columns=['symbol', 'currentTime'])
            result.append(df)
    return pd.concat(result)

# tick starts from 0 
def getArray(df, tick):
    subDataFrame = getSubDataFrame(df, tick)
    # convert to np array
    data = []
    for i in range(countPerDay):
        startIndex = i * symbolCount
        endIndex = (i+1) * symbolCount 
        if startIndex >= len(subDataFrame):
            return None
        data.append(subDataFrame[startIndex : endIndex].to_numpy().tolist())
    return data

def getSubDataFrame(df, tick):
    oneSetCount = countPerDay * symbolCount
    startIndex = tick * oneSetCount
    endIndex = (tick + 1) * oneSetCount
    return df[startIndex:endIndex]

if __name__ == '__main__':
    startDate, endDate = '20210831', '20210930'
    memoryBefore = getMemoryUsed()
    
    df = readToDataFrame(startDate, endDate, dropColumns=False)
    print(len(df))
    
    memoryAfter = getMemoryUsed()
    print(f'memory used {memoryAfter - memoryBefore}M')

    # read first 85 record 
    data = getArray(df, 0)
    print(data[0][0][0], data[0][0][1])
    print(data[0][122][0], data[0][122][1])
    print(data[84][0][0], data[84][0][1])
    print(data[84][122][0], data[84][122][1])

    data = getArray(df, 1)
    print(data[0][0][0], data[0][0][1])
    print(data[0][122][0], data[0][122][1])
    print(data[84][0][0], data[84][0][1])
    print(data[84][122][0], data[84][122][1])

    print(type(data))
    print('data shape: ', np.array(data).shape)

    assert getArray(df, 100) is None
