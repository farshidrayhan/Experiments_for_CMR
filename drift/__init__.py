import sklearn.datasets
import sklearn.tree
import collections
import numpy

from sklearn.model_selection import train_test_split

clf = sklearn.tree.DecisionTreeClassifier(random_state=42)
iris = sklearn.datasets.load_iris()


x_train ,  x_test , y_train  ,  y_test=  train_test_split(iris.data,iris.target,test_size=.3)
# x_test , y_test = iris.data[90:] , iris.target[90:]

number_of_features = len(x_train[0])

print("### train phase ###",end='\n\n')

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

counter = 0
# for i in tree_structure:
#     print(tree_structure[i][1])
#     counter += 1
#
# print(counter)
tree_leafs_unique = list(set(clf.apply(x_train)))   # leafs contains the leaf nodes of the tree
tree_leafs = clf.apply(x_train)   # leafs contains the leaf nodes of the tree
# print(list(tree_leafs))
print('nodes who are leaf ',list( tree_leafs_unique))


centroid_lst = []

leaf_centroids = {}

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

        avg =avg/counter
        # print()
        # print(avg)
        temp_centroid[y] = avg
        # print(y)

        # print()
    leaf_centroids[i] = temp_centroid
    # print(temp_centroid)
    # break
#
for i in leaf_centroids:
    print(i , " " , leaf_centroids[i])

print()
print("### Test Phase ###",end='\n\n')
# clf.fit(x_test, y_test)
tree_leafs_unique = list(set(clf.apply(x_test)))   # leafs contains the leaf nodes of the tree
tree_leafs = clf.apply(x_test)   # leafs contains the leaf nodes of the tree
print(tree_leafs)

# for i in tree_leafs:
#     print(samples[i])
#     counter += 1
#
# print(counter)
