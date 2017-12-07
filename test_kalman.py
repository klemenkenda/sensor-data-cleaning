# includes
import numpy as np
import scipy.stats
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
    bic1 = model_fit.bic
    
    
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
    bic2 = modelu_fit.bic
    modelu_fit
    
    print("BIC - raw:", bic1, " BIC - clean:", bic2)
    
    #plt.plot(uts, label="Clean data")
    #plt.plot(ts, label="Raw data")
    #plt.legend(loc='lower right', prop={'size': 9})
    #plt.savefig('c' + str(sensorId) + ".png")
    #plt.close();

if __name__ == '__main__':
    p, q, r = LearnOptimalParameters(95081)
    CompareCleanRaw(95081, p, q, r)