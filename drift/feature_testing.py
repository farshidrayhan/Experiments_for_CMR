import pandas
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder



df = pandas.read_csv('abalone19.txt',header=None)

# df = pandas.DataFrame(df)

# print(df.iloc[1])

# encoder = OneHotEncoder()
# encoder.fit(df)
# print(df.shape[1])

df['label'] = df[df.shape[1] - 1]

df.drop([df.shape[1] - 2], axis=1, inplace=True)

labelencoder = LabelEncoder()
df['label'] = labelencoder.fit_transform(df['label'])

X = np.array(df.drop(['label'], axis=1))
y = np.array(df['label'])

print(X)
df = pandas.DataFrame(X)
#
df = pandas.get_dummies(df , columns=[0,1,2])
# df = pandas.get_dummies(df)
print(df)
#
# X= []
# y = []

# for l in df:
    # last_index = len(l.v) - 1
    # X.append(l[0:last_index])
    # y.append(l[last_index])
    # print(l)