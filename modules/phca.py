# module containing the PHCA and Multi-Level PHCA models

from sklearn.cluster import KMeans
from tqdm import tqdm
import numpy as np
import ripser
import copy
import math

import warnings
warnings.filterwarnings('ignore')

class PHCA:
    def __init__(self, dim=0):
        self.dim = dim
    
    def fit(self, X_train, y_train):
        self.byClass_data = by_class_data(X_train, y_train)
        self.PD_byClass = get_persistence_diagram(self.byClass_data, self.dim)
    
    def predict(self, X_test):
        y_pred = []
        for i,x_new in enumerate(tqdm(X_test)):
            self.byClass_data_plus = update_data(self.byClass_data, x_new)
            self.PD_byClass_plus = get_persistence_diagram(self.byClass_data_plus, self.dim)
            self.selected_class = class_category_selector(self.PD_byClass, self.PD_byClass_plus, self.dim)
            y_pred.append(self.selected_class)
        return y_pred
    
class MultiLevelPHCA:
    def __init__(self, num_categories, dim=0, method='kmeans'):  # try standard or natural
        self.dim = dim
        self.numcat = num_categories
        self.method = method

    def fit(self, X_train, y_train):
        self.byClass_data = by_class_data(X_train, y_train)
        self.byCategory_data, self.clss_category = by_category_data(X_train, y_train, self.byClass_data, self.numcat, self.method)
        self.PD_byClass = get_persistence_diagram(self.byClass_data, self.dim)
        self.PD_byCategory = get_persistence_diagram(self.byCategory_data, self.dim)

    def predict(self, X_test):
        y_pred = []
        for i,x in enumerate(tqdm(X_test)):
            # category selection
            self.byCategory_data_plus = update_data(self.byCategory_data, x)
            self.PD_byCategory_plus = get_persistence_diagram(self.byCategory_data_plus, self.dim)
            self.selected_category = class_category_selector(self.PD_byCategory, self.PD_byCategory_plus, self.dim)

            # class selection
            self.selected_classes = self.clss_category[self.selected_category]  # classes within selected category
            self.byClass_data_subset = {clss: self.byClass_data[clss] for clss in self.selected_classes}  # selected classes
            self.PD_byClass_subset = {clss: self.PD_byClass[clss] for clss in self.selected_classes}  # PDs of selected classes
            self.byClass_data_plus = update_data(self.byClass_data_subset, x)  # add new test data to each class
            self.PD_byClass_plus = get_persistence_diagram(self.byClass_data_plus, self.dim)
            self.selected_class = class_category_selector(self.PD_byClass_subset, self.PD_byClass_plus, self.dim)
            y_pred.append(self.selected_class)
        return y_pred


######################### HELPER FUNCTIONS ##############################
def by_class_data(X, y):
    byClass_data = {clss: [] for clss in set(y)}
    for i in range(len(X)):
        byClass_data[y[i]].append(X[i])
    return byClass_data

def get_persistence_diagram(data, dim):
    return {key: ripser.ripser(np.array(data[key]), maxdim=dim)['dgms'] for key in data.keys()}

def update_data(data:dict, x_new:list):
    data_plus = copy.deepcopy(data)
    for clss in data_plus.keys():
        data_plus[clss].append(x_new)
    return data_plus

def get_lifespan(PD:list, dim:int):
    total_lifespan = 0
    for d in range(0, dim+1):
        births, deaths = PD[d][:,0], PD[d][:,1]
        births, deaths = births[np.isfinite(deaths)], deaths[np.isfinite(deaths)]  # remove inf values
        total_lifespan += np.sum(deaths) - np.sum(births)
    return total_lifespan

def class_category_selector(orig_PD:dict, new_PD:dict, dim:int):
    scores = {}
    for clss in orig_PD.keys():
        orig_lspan, new_lspan = get_lifespan(orig_PD[clss], dim), get_lifespan(new_PD[clss], dim)
        scores[clss] = abs(new_lspan - orig_lspan)
    return min(scores, key=scores.get)

def by_category_data(X_train, y_train, byClass_data, num_categories, method):
    if method == 'standard':
        class_inds = np.sort(np.unique(y_train))
        clss_cat = class_inds.reshape(num_categories, math.ceil(len(class_inds)/num_categories))  # 0 to numclass w shape (num_categories, numclass_per_category)
    elif method == 'natural':
        clss_cat = [
            [0,1,2,3,4,5,6,7,8,9], [10,11,12,13,14,15,16,17,18,19], [20,21,22,23,24,25,26,27,28,29],
            [30,31,32,33,34,35,36,37,38,39,40,41], [42,43,44,45,46,47,48,49,50,51], [52,53,54,55,56,57,58,59,60,61],
            [62,63,64,65,66,67,68,69,70,71], [72,73,74,75,76,77,78,79,80,81,82,83,84], [85,86,87,88,89,90,91,92,93,94],
            [95,96,97,98,99,100,101,102,103,104]  # natural categories from FSL-105 dataset
        ]
    else:
        centroids = np.array([np.mean(np.array(byClass_data[key]), axis=0) for key in byClass_data.keys()])
        kmeans = KMeans(n_clusters=num_categories, max_iter=100)
        kmeans.fit(centroids)
        clss_cat = [[i for i,clss in enumerate(kmeans.labels_) if clss==idx] for idx in np.arange(num_categories)]

    byCategory_data = {cat: [] for cat in np.arange(len(clss_cat[:num_categories]))}
    for i in byCategory_data.keys():
        indices = [idx for idx in range(len(y_train)) if y_train[idx] in clss_cat[i]]
        byCategory_data[i].extend([X_train[idx] for idx in indices])
    return byCategory_data, clss_cat[:num_categories]