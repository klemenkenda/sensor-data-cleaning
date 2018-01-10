import pandas as pd
import numpy as np
import urllib.request
from time import time
import sys, traceback;

# API URL to underground water levels in Slovenia
def LoadData(StationId):
    url = "http://atena.ijs.si:8080/CollectorAPIServer/undergroundWater?station_id=" + str(StationId)
    jsonStr = urllib.request.urlopen(url).read().decode('utf-8')
    df = pd.read_json(jsonStr)
    # format feature vector appropriately
    X = np.array(df['Value'])
    X = X.reshape(X.shape[0],-1)
    # compute mean standard deviation
    ts = pd.Series(df['Value'])
    # converting unix timestamp to date-time object
    df['Date'] = df['LastUpdatedEpoch']
    #df['Date'] = pd.to_datetime(df['Date'], unit='ms')
    
    # remove unneccessary fields
    df.drop('LastUpdated', 1, inplace=True)
    df.drop('LastUpdatedEpoch', 1, inplace=True)
    df.drop('Region_id', 1, inplace=True)
    df.drop('Region_name', 1, inplace=True)
    df.drop('Station_id', 1, inplace=True)
    df.drop('Station_name', 1, inplace=True)
    df.drop('SystemCodeNumber', 1, inplace=True)

    if len(df) < 100:
        print("Too few data!")
        sys.exit(0)
    
    # calculate average noise
    noise = ts.rolling(window = 10, center = False).std().median()
    return df, X, noise

def IntroduceNoise(X, noise, noiseAmp):
    np.random.seed(1)
    # introduce noise
    y = []
    for i in range(0, len(X)):
        error = 0
        if (np.random.uniform() < 0.05):
            o = np.random.normal(0, noiseAmp)
            if (abs(o) > 5 * noise):
                X[i] = X[i] + o
                #print(X, o)
                error = 1
        y.append(error)
        
    y = np.array(y)
    return X, y



# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
