#libraries
from sklearn import svm
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import top_k_accuracy_score as top_k
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestCentroid
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from sklearn import svm
from sklearn.model_selection import cross_val_predict
from itertools import combinations
from tqdm import tqdm
import random
import numpy as np
from scipy.special import softmax
import torchvision
from sklearn.metrics import confusion_matrix
import torch
from tqdm import tqdm
import pickle
import sys
import os
import functools

# local vars
sys.path.append(os.getenv('BONNER_ROOT_PATH'))
from image_tools.loading import load_places_cat_labels
from config import CACHE


class NearestCentroidDistances(NearestCentroid):
    def predict_distances(self, X):
        check_is_fitted(self)
        X = check_array(X, accept_sparse='csr')
        distances = pairwise_distances(X, self.centroids_, metric=self.metric)
        return distances
    

    
def prototype_performance(X_train, y_train, X_test, y_test):
        model = NearestCentroidDistances()
        model.fit(X_train, y_train)
        y_pred = model.predict_distances(X_test)
        y_pred = softmax(-y_pred, axis=1)   
        y_pred = np.argmax(y_pred, axis=1)

        return accuracy_score(y_test, y_pred)


    
def get_Xy(data):
    
    cat_labels = load_places_cat_labels()
    images = list(cat_labels.keys())

    data = data.x.values
    labels = np.array([cat_labels[i] for i in images])
    
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)
    
    return data, labels



def logistic_regression(X_train, y_train, X_test, y_test):
    
    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    return clf.score(X_test, y_test)


def cv_performance(X, y, cat_labels, clf = 'logistic', num_folds=5):
    
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    labels = np.array(list(cat_labels.keys()))
    categories = np.array(list(cat_labels.values()))

    accuracy = []
    
    for train_index, test_index in skf.split(labels, categories):
        
        X_train, y_train = X[train_index,...], y[train_index,...]
        X_test, y_test = X[test_index,...], y[test_index,...]

        if clf == 'prototype':
            accuracy_score = prototype_performance(X_train, y_train, X_test, y_test)
            accuracy.append(accuracy_score)
        
        elif clf == 'logistic':
            accuracy_score = logistic_regression(X_train, y_train, X_test, y_test)
            accuracy.append(accuracy_score)    
    
    return sum(accuracy)/len(accuracy) 



def cache(file_name_func):

    def decorator(func):
        
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):

            file_name = file_name_func(*args, **kwargs) 
            cache_path = os.path.join(CACHE, file_name)
            
            if os.path.exists(cache_path):
                return 
            
            result = func(self, *args, **kwargs)
            with open(cache_path,'wb') as f:
                pickle.dump(result,f)
            print('classification results are saved in cache')
            return 

        return wrapper
    return decorator



# class PairwiseClassification():
    
#     def __init__(self):
        
#         if not os.path.exists(os.path.join(CACHE,'classification')):
#             os.mkdir(os.path.join(CACHE,'classification'))

#     @staticmethod
#     def cache_file(iden, data):
#         return os.path.join('classification',iden)

   
#     @cache(cache_file)
#     def get_performance(self, iden, data):
        
#         performance_dict = {}

#         for cat_1, cat_2 in tqdm(combinations(CAT_SUBSET, 2),
#                                  total=(len(CAT_SUBSET)*(len(CAT_SUBSET)-1)//2)):
            
#                 X, y = get_Xy(data, [cat_1, cat_2])
#                 performance_dict[(cat_1, cat_2)] = cv_performance(X, y)
                
#         return performance_dict


def normalize(X):
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    return X

