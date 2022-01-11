import os

import re

def stringToAscii(s):
    return ''.join([str(ord(x)) for x in s.upper()])

def ascToString(s):
    chars = re.findall('..?', s)
    return ''.join([chr(int(x)) for x in chars])

"""TECL,2021-11-09 10:00:00"""
def encode(date, symbol):
    dateText = date.replace("-", "")
    dateText = dateText.replace(" ", "")
    dateText = dateText.replace(":", "")
    symbolNumber = stringToAscii(symbol)
    return int(dateText + symbolNumber)

def decode(textNumber):
    text = str(textNumber)
    dateText = text[0:12]
    symbolNumber = text[14:]
    symbol = ascToString(symbolNumber)
    return f"{dateText} {symbol}"


def debug(txt): 
    if os.getenv('mymuzero_env') == 'dev':
        print(txt)