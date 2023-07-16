import torch
from sklearn import svm
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import top_k_accuracy_score as top_k
from tqdm import tqdm
import numpy as np 
import random

def create_splits(n: int, num_folds: int = 5, shuffle: bool = True): 
    
    random.seed(0)
    if shuffle:
        indices = np.arange(0,n)
        random.shuffle(indices)
    else:
        indices = np.arange(0,n)

    x = np.array_split(indices, num_folds)
    return x



def train(features, labels, estimator_type, shuffle = True, num_folds=10):

    
    splits = create_splits(n = len(features), shuffle = shuffle, num_folds=num_folds)
    top_1, top_5 = [], []

        
    for indices_test in tqdm(splits):
        
        if estimator_type == 'svm':
            classifier = svm.SVC(probability=True)

        if estimator_type == 'logistic':
            classifier = LogisticRegression()
        
        indices_train = np.setdiff1d(np.arange(0, len(features)), np.array(indices_test))
        X_train, y_train = features[indices_train,...], labels[indices_train,...]
        X_test, y_test = features[indices_test,...], labels[indices_test,...]
        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict_proba(X_test)
    
        top_1.append(top_k(y_test, y_pred, k=1))
        top_5.append(top_k(y_test, y_pred, k=5))
        
        
    return sum(top_1)/num_folds, sum(top_5)/num_folds
