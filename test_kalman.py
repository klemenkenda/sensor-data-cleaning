# includes
import numpy as np
import scipy.stats
import math
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

import matplotlib.pyplot as plt

from joblib import Parallel, delayed
import multiprocessing
from kalmanaotsclassifier import KalmanAOTSClassifier

from utilities import *

from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMAResults
import sklearn.metrics


def LearnOptimalParameters(stationId, n_iter = 20, size = 3000):
    param_dist = {
        "p_fact": [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1., 3., 10, 30],
        "r_fact": [0.003, 0.01, 0.03, 0.1, 0.3, 1., 3., 10],
        "q_fact": [0.00001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
    }

    
    df, X, noise = LoadData(stationId)    
    X, y = IntroduceNoise(X, noise, 40 * noise)    

    clf2 = KalmanAOTSClassifier()
    n_iter_search = 20
    search = RandomizedSearchCV(clf2, param_distributions=param_dist, n_iter=n_iter_search, cv=2, n_jobs=2)
    #search = GridSearchCV(clf2, param_grid=param_dist, cv=2, n_jobs = 4)
    search.fit(X[0:size], y[0:size])

    results = search.cv_results_

    report(results)

    candidates = np.flatnonzero(results['rank_test_score'] == 1)
    for candidate in candidates:
        params = results['params'][candidate]
    print("Score: ", results['mean_test_score'][candidate])
    #print("Precision: ", results['mean_test_precision'][candidate])
    print("all keys")
    
    #for key, value in results.iteritems():
    #    print(key, value)

    return params['p_fact'], params['q_fact'], params['r_fact']

def CleanData(stationId, p, q, r):
    df, X, noise = LoadData(stationId)
    
    if len(df) == 0:
        return 0, 0, noise, [], []
        
    y = np.zeros(len(X))
    
    clftrue = KalmanAOTSClassifier(p_fact = p, q_fact = q, r_fact = r)    
    clftrue.fit(X, y)

    pred = clftrue.predict(X)
    yt = clftrue.tspredictions

    return len(df), np.array(pred).sum(), noise, pred, yt

def CompareCleanRaw(sensorId, p, q, r):
    # load data
    df, X, noise = LoadData(sensorId)
    # convert data frame in to timeseries
    df['Date'] = pd.to_datetime(df['Date'], unit='ms')
    df.set_index('Date', inplace=True)
    # convert to timeseries
    startPlot = 50
    ts = pd.Series(df['Value'][startPlot:])
    
    # fit model
    model = ARIMA(ts, order=(1,1,0))
    model_fit = model.fit(disp=0)
    rmse1 = math.sqrt(model_fit.sigma2)
    
    
    # clean data
    # clean data and make clean set
    length, errors, noise, pred, yt = CleanData(sensorId, p, q, r)
    print("Sensor", sensorId)
    print("Errors: ", errors, " (", errors/length, ")")
    pred = pred[startPlot:]
    yt = yt[startPlot:]
    
    uts = ts.copy()
    for i in range(1, len(yt)):
        if (pred[i] == 1):
            uts[i] = yt[i]
            
    # fit model
    modelu = ARIMA(uts, order=(1,1,0))
    modelu_fit = modelu.fit(disp=0)
    rmse2 = math.sqrt(modelu_fit.sigma2)
    #modelu_fit
    
    print("RMSE - raw:", rmse1, " RMSE - clean:", rmse2)
    
    with open("results.csv", "a") as f:
        f.write("%s;%f;%f;%f;%f;%f;%d;%d;%d" % (sensorId, p, q, r, rmse1, rmse2, rmse2 < rmse1, errors, length))

    #plt.plot(uts, label="Clean data")
    #plt.plot(ts, label="Raw data")
    #plt.legend(loc='lower right', prop={'size': 9})
    #plt.savefig('c' + str(sensorId) + ".png")
    #plt.close();

if __name__ == '__main__':
    print("Loading all sensors ...")
    # API URL to underground water levels Slovenia
    # list all available sensors
    url = "http://atena.ijs.si:8080/CollectorAPIServer/undergroundWater?getStations"
    jsonStr = urllib.request.urlopen(url).read().decode('utf-8')
    sources = pd.read_json(jsonStr)

    # go through all sensors and run algorithm for them
    allNodes = []

    l = len(sources)
    l = 2

    with open("results.csv", "w") as f:
        f.write("SensorId;p;q;r;RMSE1;RMSE2;better clean fit;Errors;Length;Length of DF;avgWindow;noise\n")

    for i in range(0, l):
        try:
            df, X, noise = LoadData(sources["Station_id"][i])
            avgWindow = (df['Date'][len(df) - 1] - df['Date'][0]) / 1000 / 60 / 60 / 24 / len(df)
            #if (avgWindow < 1.05) and (noise < 0.2):
            print("Chosen", sources["Station_id"][i])
            # p, q, r = LearnOptimalParameters(sources["Station_id"][i])
            # CompareCleanRaw(sources["Station_id"][i], p, q, r)
            CompareCleanRaw(sources["Station_id"][i], 1, 1, 1)

            with open("results.csv", "a") as f:
                f.write("%d;%f;%f;\n" % (len(df), avgWindow, noise))
            #else:
            #    print("Rejected", sources["Station_id"][i])
        except Exception as inst:
            print(type(inst))    # the exception instance
            print(inst.args)     # arguments stored in .args


    #p, q, r = LearnOptimalParameters(95081)
    #CompareCleanRaw(95081, p, q, r)