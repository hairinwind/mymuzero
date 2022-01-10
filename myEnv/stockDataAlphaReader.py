from datetime import date, datetime, timedelta
from itertools import count

import numpy as np
import os
import pandas as pd

countPerDay = 9 # as it is resampled every 15 minutes, the countPerDay was changed from 85 to 29
symbolCount = 120
signalCount = 11

"""
- dropColumns shall be True, False is for testing purpose
"""
def readToDataFrame(startDate, endDate):    
    file = os.path.join('data/working', 'hourly_data.csv')
    if os.path.isfile(file):
        # print(f'read file for {day}')
        df = pd.read_csv(file)
        df = df[(df['date']>=startDate) & (df['date']<=endDate)]
        df.sort_values(by=['time','symbol'], inplace=True)
        return df

def getArray(df, tick, dropColumn=True):
    subDataFrame = getSubDataFrame(df, tick)

    # convert symbol to int or drop symbol?
    # convert date to int
    if dropColumn:
        subDataFrame = subDataFrame.drop(columns=['symbol', 'time', 'date', 'hourTime'])
    
    # convert to np array
    data = []
    for i in range(countPerDay):
        startIndex = i * symbolCount
        endIndex = (i+1) * symbolCount 
        if endIndex > len(subDataFrame):
            print(endIndex, len(subDataFrame))
            return None
        data.append(subDataFrame[startIndex : endIndex].to_numpy().tolist())
    if len(data) == 0:
        return None
    return data

"""
tick starts from 0
"""
def getSubDataFrame(df, tick):
    startIndex = tick * symbolCount
    endIndex = startIndex + countPerDay * symbolCount
    return df[startIndex:endIndex]


if __name__ == "__main__":
    # # the code to add missed quotes from either previous one or next one.
    # hours = [
    #     '09:35:00', '10:00:00', '11:00:00', '12:00:00', '13:00:00', '14:00:00', '15:00:00', '16:00:00', '16:05:00'
    # ]
    # data = pd.read_csv('data/training/hourly_data.csv')
    # df1 = data.groupby(['symbol','date']).agg({'symbol':'first', 'date':'first','time':'count'})
    # df2 = df1[df1['time']!=9]
    # print(df2)
    # count = 0
    # for index, row in df2.iterrows():
    #     print(row['symbol'], row['date'], row['time'])
    #     sub_df = data[(data['symbol'] == row['symbol']) & (data['date'] == row['date'])]
    #     hoursFound = sub_df['hourTime'].tolist()
    #     missedTimes = [x for x in hours if x not in hoursFound]
    #     for missedTime in missedTimes:
    #         quotesBeforeMissed = sub_df[data['hourTime']<missedTime]
    #         if (len(quotesBeforeMissed) > 0):
    #             replaceQuote = quotesBeforeMissed.tail(1).copy()
    #         else:
    #             replaceQuote = sub_df[data['hourTime']>missedTime].head(1).copy()
    #         replaceQuote['hourTime'] = missedTime
    #         replaceQuote['time'] = replaceQuote['date'] + " " + replaceQuote["hourTime"]
    #         data = data.append(replaceQuote, ignore_index=True)
    #     count += 1
    # data.to_csv('data/training/hourly_data.csv', index=False)
        

    df = readToDataFrame('2021-11-17','2021-12-30')
    arrayData0 = getArray(df, 0, False)
    print(np.array(arrayData0).shape)
    arrayData1 = getArray(df, 1, False)
    print(np.array(arrayData1).shape)
    
    array1 = arrayData0[1][0]
    array2 = arrayData1[0][0]
    assert np.array_equal(array1, array2)

    print(array2)
    print(getArray(df, 2, False)[0][0])