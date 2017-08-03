import re
import random

import numpy
from imblearn.under_sampling import RandomUnderSampler
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn import svm, metrics
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import sklearn
from sklearn.model_selection import cross_val_predict, GridSearchCV
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sympy.functions.special.gamma_functions import gamma
from sklearn.metrics import classification_report, average_precision_score

total_matrix = [[]]

total_number_of_simulations = 50
print("Number of simulations " , total_number_of_simulations,end='\n\n')
dataset_list = ['glass0.txt' ,'ecoli-0-3-4_vs_5.txt','glass-0-1-5_vs_2.dat.txt','yeast-1_vs_7.txt']

for dataset in dataset_list:

    feature_list_of_all_instances = []
    class_list_of_all_instances = []
    total_matrix = []
    k_fold = 5
    count_for_number_of_instances = 0
    index = 0


    print("For dataset " , dataset)
    with open(dataset, 'r') as file_read:

        for x in file_read:
            index += 1
            if len(x) <= 10:
                break
            l = x.rstrip("\n").split(",")
            last_index = len(l) - 1
            if l[last_index] == 'negative' or l[last_index] == ' negative':
                l[last_index] = 0
            else:
                l[last_index] = 1
            l = list(map(float, l))
            total_matrix.append(l)

    for l in total_matrix:
        last_index = len(l)-1

        feature_list_of_all_instances.append(l[0:last_index])
        class_list_of_all_instances.append(l[last_index])

    kf = StratifiedKFold(n_splits=k_fold, shuffle=True)
    under_sample = RandomUnderSampler()

    top_avg_roc = 0
    for simulation in range (0,total_number_of_simulations):

        avg_roc = 0

        for train_set_indexes, test_set_indexes in kf.split(feature_list_of_all_instances, class_list_of_all_instances):

            temp_train_feature_list = []
            temp_train_class_list = []
            for index in train_set_indexes:
                temp_train_feature_list.append(feature_list_of_all_instances[index])
                temp_train_class_list.append(class_list_of_all_instances[index])

            temp_test_feature_list = []
            temp_test_class_list = []
            for index in test_set_indexes:
                temp_test_feature_list.append(feature_list_of_all_instances[index])
                temp_test_class_list.append(class_list_of_all_instances[index])

            sampler = RandomUnderSampler()
            temp_train_feature_list, temp_train_class_list = sampler.fit_sample(temp_train_feature_list,
                                                                                temp_train_class_list)


            # clf = KMeans(n_clusters=20,max_iter=200,algorithm='auto',n_jobs=4)
            #
            # clf.fit(temp_train_feature_list,temp_train_class_list)
            #
            # predicted = clf.predict(temp_test_feature_list)
            #
            # avg_roc += sklearn.metrics.roc_auc_score(temp_test_class_list, predicted)

        if top_avg_roc < avg_roc/k_fold:
            top_avg_roc = avg_roc/k_fold

    print("     Best average Roc score for knn", top_avg_roc)

    top_avg_roc = 0
    for simulation in range (0,total_number_of_simulations):

        avg_roc = 0

        for train_set_indexes, test_set_indexes in kf.split(feature_list_of_all_instances, class_list_of_all_instances):

            temp_train_feature_list = []
            temp_train_class_list = []
            for index in train_set_indexes:
                temp_train_feature_list.append(feature_list_of_all_instances[index])
                temp_train_class_list.append(class_list_of_all_instances[index])

            temp_test_feature_list = []
            temp_test_class_list = []
            for index in test_set_indexes:
                temp_test_feature_list.append(feature_list_of_all_instances[index])
                temp_test_class_list.append(class_list_of_all_instances[index])

            sampler = RandomUnderSampler()
            temp_train_feature_list, temp_train_class_list = sampler.fit_sample(temp_train_feature_list,
                                                                                temp_train_class_list)


            clf = DecisionTreeClassifier(max_depth=17,min_samples_split=3)

            clf.fit(temp_train_feature_list,temp_train_class_list)

            predicted = clf.predict(temp_test_feature_list)

            avg_roc += sklearn.metrics.roc_auc_score(temp_test_class_list, predicted)

        if top_avg_roc < avg_roc/k_fold:
            top_avg_roc = avg_roc/k_fold

    print("     Best average Roc score for Decision Tree", top_avg_roc)

