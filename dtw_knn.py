import numpy as np
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import os
import pandas as pd
import sklearn
from sklearn import preprocessing
import numpy as np
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
#
#toy dataset
# X = np.random.random((100,10))
# print(X.shape)
# y = np.random.randint(0,2, (100))
# print(y.shape)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#
# print(type(X_train))
# print(y_train)
# print(X_test.shape)




data_path= r'/code/DTW/data_ceshiji_train'

# data_path= r'/code/tsc/dataset/data/data_train'


y_train = pd.read_csv(os.path.join(data_path, 'train_labels.csv'))
y_test = pd.read_csv(os.path.join(data_path, 'test_labels.csv'))

X_train = pd.read_csv(os.path.join(data_path, 'train_new.csv'))
X_test = pd.read_csv(os.path.join(data_path, 'test_new.csv'))

nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

# transform the labels from integers to one hot vectors
enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
y_train = enc.transform(y_train.values.reshape(-1, 1)).toarray()
y_test = enc.transform(y_test.values.reshape(-1, 1)).toarray()

# save orignal y because later we will use binary(lablel转化为2进制)
y_true = np.argmax(y_test, axis=1)

if len(X_train.shape) == 2:  # if univariate
    # add a dimension to make it multivariate with one dimension
    x_train = X_train.values.reshape((X_train.shape[0], X_train.shape[1]))
    x_test = X_test.values.reshape((X_test.shape[0], X_test.shape[1]))

print(type(X_train))
print(X_train.shape)
print(y_train)


#custom metric
def DTW(a, b):   
    an = a.size
    bn = b.size
    pointwise_distance = distance.cdist(a.reshape(-1,1),b.reshape(-1,1))
    cumdist = np.matrix(np.ones((an+1,bn+1)) * np.inf)
    cumdist[0,0] = 0

    for ai in range(an):
        for bi in range(bn):
            minimum_cost = np.min([cumdist[ai, bi+1],
                                   cumdist[ai+1, bi],
                                   cumdist[ai, bi]])
            cumdist[ai+1, bi+1] = pointwise_distance[ai,bi] + minimum_cost

    # print(cumdist[an, bn])
    return cumdist[an, bn]#累积概率？





#train
#n_neighbors : 表示选择距离最近的K个点来投票的数量。
parameters = {'n_neighbors':[1]}
clf = GridSearchCV(KNeighborsClassifier(metric=DTW), parameters, cv=3, verbose=1)
clf.fit(X_train, y_train)



#evaluate
y_pred = clf.predict(X_test)
print(y_pred)
print(classification_report(y_test, y_pred))
