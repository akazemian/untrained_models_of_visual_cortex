from sklearn import svm
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import top_k_accuracy_score as top_k
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestCentroid
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import svm
from sklearn.model_selection import cross_val_predict
from tqdm import tqdm
import random
import numpy as np
from scipy.special import softmax
import torchvision
from sklearn.metrics import confusion_matrix
import torch
from tqdm import tqdm
import pickle


from tools.loading import *
from analysis.encoding_model_analysis.tools.utils import get_activations_iden, get_scores_iden
from analysis.encoding_model_analysis.tools.extractor import Activations
import config




def create_splits(n: int, num_folds: int = 5, shuffle: bool = True): 
    
    random.seed(0)
    if shuffle:
        indices = np.arange(0,n)
        random.shuffle(indices)
    else:
        indices = np.arange(0,n)

    x = np.array_split(indices, num_folds)
    return x



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


    
def get_Xy(data, categories):
    
    cat_labels = load_places_cat_labels()
    cat_labels_subset = {k: v for k, v in cat_labels.items() if v in categories}
    images_subset = list(cat_labels_subset.keys())

    data_subset = data.sel(stimulus_id = images_subset).x.values
    labels_subset = np.array([cat_labels_subset[i] for i in images_subset])
    
    encoder = LabelEncoder()
    labels_subset = encoder.fit_transform(labels_subset)
    
    return data_subset, labels_subset




def cv_performance(X, y, num_folds=5):
    
    splits = create_splits(n = len(X), shuffle = True, num_folds=num_folds)
    accuracy = []
    
    for indices_test in splits:

        indices_train = np.setdiff1d(np.arange(0, len(X)), np.array(indices_test))
        
        X_train, y_train = X[indices_train,...], y[indices_train,...]
        X_test, y_test = X[indices_test,...], y[indices_test,...]

        accuracy_score = prototype_performance(X_train, y_train, X_test, y_test)
        accuracy.append(accuracy_score)
    
    return sum(accuracy)/len(accuracy) 







def train(features, labels, estimator_type, shuffle = True, num_folds=10):

    
    features = normalize(features)
    
    
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




def normalize(X):
    return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))