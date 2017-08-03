# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 15:36:52 2017

@author: Sajid
"""

import pandas as pd
import numpy as np


from sklearn.cluster import KMeans

from sklearn.tree import DecisionTreeClassifier


from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import Normalizer


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# df = pd.read_csv('F:\\CMR sir thesis\\ecoli\\ecoli.dat',skiprows=12,header=None)
#
# df = pd.read_csv('F:\\CMR sir thesis\\pima\\pima.dat',skiprows=13,header=None)
#
# df = pd.read_csv('F:\\CMR sir thesis\\banana\\banana.dat',skiprows=7,header=None)

#just set the path to the txt file here
dataset_list = ['/home/farshid/Desktop/thyroid.txt']

for dataset in dataset_list:

    print("Dataset ",dataset)

    df = pd.read_csv(dataset,header=None)


    df['label'] = df[df.shape[1] - 1]

    df.drop([df.shape[1]-2],axis=1,inplace=True)

    labelencoder = LabelEncoder()
    df['label'] = labelencoder.fit_transform(df['label'])

    X = np.array(df.drop(['label'],axis=1))
    y = np.array(df['label'])

    normalization_object = Normalizer()
    X = normalization_object.fit_transform(X)

    number_of_folds = 10

    best_clustered_trees_average_auc = 0
    best_one_tree_average_auc =0
    best_cluster = 0


    for simullation in range(0,30):

        print("number of simulation " ,simullation +1)
        skf = StratifiedKFold(n_splits=number_of_folds,shuffle=True,random_state=simullation)

        for n in range(2,50):
            number_of_clusters = n
            trees = {}
            all_auc_with_clustered_trees = []
            all_auc_with_one_tree = []


            for train_index, test_index in skf.split(X, y):
                X_train = X[train_index]
                X_test = X[test_index]

                y_train = y[train_index]
                y_test = y[test_index]

                #optimize for number of clusters here
                kmeans = KMeans(n_clusters=number_of_clusters,max_iter=200,n_jobs=2)
                kmeans.fit(X_train)


                #get the centroids of each of the clusters
                cluster_centroids = kmeans.cluster_centers_

                #get the points under each cluster
                points_under_each_cluster =   {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}

                for i in range(number_of_clusters) :
                    classifier = DecisionTreeClassifier()

                    classifier.fit(X_train[points_under_each_cluster[i]],y_train[points_under_each_cluster[i]])
                    trees[i] = classifier

                #fitting one tree on the whole train set
                classifier = DecisionTreeClassifier(max_depth=17,min_samples_split=3)
                classifier.fit(X_train,y_train)

                predictions_by_clustered_trees = []

                for instance in X_test :
                    tree_to_use = np.argmin(np.linalg.norm(cluster_centroids - instance , axis = 1))
                    # print(instance)
                    # print(tree_to_use)
            #        print(instance)
            #        break
                    point = np.array(instance)
                    prediction = trees[tree_to_use].predict([point])
                    predictions_by_clustered_trees.append(prediction[0])

                predictions_by_clustered_trees = np.array(predictions_by_clustered_trees)

                predictions_by_one_tree = classifier.predict(X_test)


                all_auc_with_clustered_trees.append(roc_auc_score(y_test,predictions_by_clustered_trees))
                all_auc_with_one_tree.append(roc_auc_score(y_test,predictions_by_one_tree))


            clustered_trees_average_auc = sum(all_auc_with_clustered_trees)/len(all_auc_with_clustered_trees)
            one_tree_average_auc = sum(all_auc_with_one_tree)/len(all_auc_with_one_tree)
            #
            # print("Clustered_trees_average_auc ",clustered_trees_average_auc)
            # print("One_tree_average_auc " , one_tree_average_auc )
            # print("for number of cluster  " , number_of_clusters)

            if best_clustered_trees_average_auc < clustered_trees_average_auc:
                best_clustered_trees_average_auc = clustered_trees_average_auc
                best_cluster = number_of_clusters

        print(" best_clustered_trees_average_auc " , best_clustered_trees_average_auc , " with cluster ",best_cluster)
            #now doing other experiments
            # print(y_test[predictions_by_clustered_trees!=predictions_by_one_tree])




