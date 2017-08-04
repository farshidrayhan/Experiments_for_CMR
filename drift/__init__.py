from random import random

import sklearn.datasets
import sklearn.tree
import collections
import numpy
import pandas
from sklearn.model_selection import train_test_split

clf = sklearn.tree.DecisionTreeClassifier(random_state=42)
iris = sklearn.datasets.load_iris()

total_matrix = pandas.DataFrame(pandas.read_csv('pageblocks.txt'))

x_train = []
x_test = []
x_test2 = []
y_train = []
y_test = []
y_test2 = []

total_matrix = numpy.ndarray.tolist(pandas.DataFrame.as_matrix(total_matrix))
count = 0
X = []
y = []
print("Total instances ", len(total_matrix))
for l in total_matrix:
    last_index = len(l) - 1
    X.append(l[0:last_index])
    y.append(l[last_index])
    count += 1
    # if count == 1000 :
    #     break
X = iris.data
y = iris.target

count = 0
for i in y:
    if i == 1 or i == 4 or i == 3 :
        if random() >= .5:
            continue
        x_train.append(X[count])
        y_train.append(y[count])
    count += 1

count = 0
for i in y:
    if i == 5 or i == 2 or i == 3:
        x_test.append(X[count])
        y_test.append(y[count])
    count += 1
count = 0
for i in y:
    if i == 1 or i == 4 or i == 3:
        x_test2.append(X[count])                # non novel
        y_test2.append(y[count])
    count += 1


# x_train ,  x_test , y_train  ,  y_test=  train_test_split(X,y,test_size=.3,random_state=13)

number_of_features = len(x_train[0])

print("### training phase ###", end='\n\n')

tree = clf.fit(x_train, y_train)
#
samples = collections.defaultdict(list)
dec_paths = tree.decision_path(x_train)

for d, dec in enumerate(dec_paths):
    for i in range(tree.tree_.node_count):
        if dec.toarray()[0][i] == 1:
            samples[i].append(d)

# print(clf.apply(iris.data))
# tree_structure = clf.tree_.__getstate__()['nodes']
# print(tree_structure)
# for i in tree_structure:
#     print(tree_structure[i][1])
#     counter += 1
#
# print(counter)
tree_leafs_unique = list(set(clf.apply(x_train)))  # leafs contains the leaf nodes of the tree
tree_leafs = clf.apply(x_train)  # leafs contains the leaf nodes of the tree
# print(list(tree_leafs))
# print('nodes who are leaf in training' ,list( tree_leafs_unique))


centroid_lst = []

leaf_centroids = {}
# print("calculating Centroids ")
for i in tree_leafs_unique:
    # for j in tree_leafs:
    #     centroid_lst[i].append(x_train[i])
    # matches = list(x for x in tree_leafs if x == i)
    # print(list(tree_leafs).index[i])
    tmp_lst = []
    for k, j in enumerate(tree_leafs):
        if j == i:
            tmp_lst.append(x_train[k])

    # numpy.asarray(tmp_lst)
    # print(tmp_lst)
    avg = 0
    temp_centroid = numpy.zeros(number_of_features)

    for y in range(number_of_features):
        counter = 0
        for j in tmp_lst:
            avg += j[y]
            # print(j[y],end=' ')
            counter += 1

        avg = avg / counter
        # print()
        # print(avg)
        temp_centroid[y] = avg
        # print(y)

        # print()
    leaf_centroids[i] = temp_centroid
    # print(temp_centroid)
    # break
#
# for i in leaf_centroids:
#     print(i , " " , leaf_centroids[i])

# print()
print("### Testing Phase ###", end='\n\n')


def test(x_tester):


    leaf_centroids_test = {}
    tree_leafs_unique = list(set(clf.apply(x_tester)))  # leafs contains the leaf nodes of the tree
    tree_leafs = clf.apply(x_tester)  # leafs contains the leaf nodes of the tree

    # print('nodes who are leaf in testing ' ,list( tree_leafs_unique))
    print()

    # print(tree_leafs)

    # for i in tree_leafs:
    #     print(samples[i])
    #     counter += 1
    #
    # print(counter)
    for i in tree_leafs_unique:
        # for j in tree_leafs:
        #     centroid_lst[i].append(x_train[i])
        # matches = list(x for x in tree_leafs if x == i)
        # print(list(tree_leafs).index[i])
        tmp_lst = []
        for k, j in enumerate(tree_leafs):
            if j == i:
                tmp_lst.append(x_tester[k])

        # numpy.asarray(tmp_lst)
        # print(tmp_lst)
        avg = 0
        temp_centroid = numpy.zeros(number_of_features)

        for y in range(number_of_features):
            counter = 0
            for j in tmp_lst:
                avg += j[y]
                # print(j[y],end=' ')
                counter += 1

            avg = avg / counter
            # print()
            # print(avg)
            temp_centroid[y] = avg
            # print(y)

            # print()
        leaf_centroids_test[i] = temp_centroid
        # print(temp_centroid)
        # break
    # print("### Comparision ###")
    for i in leaf_centroids_test:
        # print("Training centroid for node ", i , " " , leaf_centroids[i])
        # print("Testing centroid for node ",i , " " , leaf_centroids_test[i])
        distance = abs(leaf_centroids[i] * leaf_centroids[i] - leaf_centroids_test[i] * leaf_centroids_test[i])
        # print("Difference between two centroids for node ", i, " : ", numpy.sqrt(distance))
        # print()
        return distance
novel = test(x_tester=x_test)
print("distance of train and test centroids with novel class")
print(novel)
print("distance of train and test centroids with no novel class")
non_novel = test(x_tester=x_test2)
print(non_novel)
print("difference of difference ")
print(novel-non_novel)
print("sum of difference of difference ")
print(numpy.sum(novel-non_novel))
