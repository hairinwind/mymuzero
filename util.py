import os

def debug(txt): 
    if os.getenv('mymuzero_env') == 'dev':
        print(txt)