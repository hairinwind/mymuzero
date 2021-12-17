from finta import TA

import pandas as pd

def customIndicators(df):
    # calculate SMA, RSI and OBV
    df['SMA12'] = TA.SMA(df, 12)
    df['SMA20'] = TA.SMA(df, 20)
    df['RSI'] = TA.RSI(df)
    df['OBV'] = TA.OBV(df)
    df.fillna(0, inplace=True)

df = pd.read_csv('data/gmedata.csv')
customIndicators(df)
print(df.head(30))
df.to_csv('data/gmedata.csv')

