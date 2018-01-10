import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from filterpy.kalman import KalmanFilter
from pykalman import KalmanFilter as KFEM
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import unique_labels
import sklearn.metrics

class KalmanAOTSClassifier(BaseEstimator, ClassifierMixin):
        
    def __init__(self, p_fact=3., r_fact=30., q_fact=0.0000015):
        # initial factors
        self.p_fact = p_fact
        self.r_fact = r_fact
        self.q_fact = q_fact
        
        # scoring = "accuracy"
    
        self.errors = []
        self.ytspredictions = []
        self.ypredicted = []

        self.Fn = 2
        self.Hn = 2
        
        # init Kalman filter
        self.kf = KalmanFilter(self.Fn, self.Hn)        
        # initial hidden state
        self.kf.x = np.array([[ 0],[ 0]]);
        # initial transition matrix
        self.kf.F = np.array([[1., 1.], [0., 1.]]);
        # initial measurements function
        self.kf.H = np.array([[1., 0.]]);
        
    def EMoptimization(self, X):
        # init Kalman filter
        self.kf = KalmanFilter(self.Fn, self.Hn)        
        # initial hidden state
        # approximate initial state
        startx = X[0]
        startkx = X[1] - X[0]
        self.kf.x = np.array([startx, startkx]);
        # initial transition matrix
        self.kf.F = np.array([[1., 1.], [0., 1.]]);
        # initial measurements function
        self.kf.H = np.array([[1., 0.]]);
        
        # perform EM expectation minimization for Kalman Filter
        lkf = KFEM(transition_matrices = self.kf.F, observation_matrices = self.kf.H)
        # print(self.kf.F, self.kf.H)
        # flatten X for em
        Xflat = np.ndarray.flatten(X)
        #print(Xflat)
        lkf.em(Xflat, n_iter = 1)

        self.EM_initialstate = lkf.initial_state_mean
        self.EM_P = lkf.transition_covariance
        self.EM_R = lkf.observation_covariance
        self.EM_Q = lkf.initial_state_covariance
        
        
    def fit(self, X, y):
        if 'EM_P' in locals():
            print("Exists")
        else:
            #print("EM was not yet run")
            self.EMoptimization(X)
                    
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        # covariance matrix
        self.kf.P = self.EM_P * self.p_fact # kf.P *= 3;
        # state uncertainty
        self.kf.R = self.EM_R * self.r_fact # kf.R = 1;
        self.kf.Q = self.EM_Q * self.q_fact # kf.Q = np.array([[0.0001, 0.0005], [0.0005, 0.0005]])

        # intial state
        self.kf.x = self.EM_initialstate

        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self

    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"p_fact": self.p_fact, "r_fact": self.r_fact, "q_fact": self.q_fact }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self    
    
    def predict(self, X):
        print("Predict is run")
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        #X = check_array(X)
        self.ypredicted = []
        self.tspredictions = []
        
        for m in X:
            m = m[0]
            self.kf.predict()
            y = self.kf.x[0] + self.kf.x[1] * 0.01 # hardcoded step (!)
            e = self.kf.P[0, 0]
            
            error = 0
            if (m < y - e):
                self.kf.P *= 2
                error = 1
            if (m > y + e):
                self.kf.P *= 2
                error = 1
            
            self.errors.append(e)
            self.tspredictions.append(y)
            
            if error == 0:
                self.kf.update(m)
            
            #print(m, y, e, error)
            self.ypredicted.append(error)

        return self.ypredicted
    
    def returnVariance(self):
        return self.errors
    
    def returnPredictions(self):
        return self.tspredictions
    
    def score(self, X, y_true):
        ly = self.predict(X);
        score = sklearn.metrics.f1_score(y_true, ly) * pow(sklearn.metrics.precision_score(y_true, ly), 5)
        # print(sklearn.metrics.classification_report(y_true, y))
        print("Score: ", score, "    p, q, r", self.p_fact, self.q_fact, self.r_fact)
        return score