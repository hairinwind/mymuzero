from finta import TA

import pandas as pd

def customIndicators(df):
    # calculate SMA, RSI and OBV
    df['SMA12'] = TA.SMA(df, 12)
    df['SMA20'] = TA.SMA(df, 20)
    df['RSI'] = TA.RSI(df)
    df['OBV'] = TA.OBV(df)
    df.fillna(0, inplace=True)

def readAndSave(fileName):
    df = pd.read_csv(fileName)
    customIndicators(df)
    print(df.head(30))
    df.to_csv(fileName)

# readAndSave('data/gmedata.csv')
readAndSave('data/evaluate/gme_evaluate_data.csv')