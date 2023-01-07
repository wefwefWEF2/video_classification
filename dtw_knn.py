import os
import pandas as pd
import sklearn
from sklearn import preprocessing
import numpy as np
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


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

    return cumdist[an, bn]






def create_classifier(data_path):
    # x_train, x_test, y_train, y_test = train_test_split(train, train_labels)

    y_train = pd.read_csv(os.path.join(data_path, 'train_labels.csv'))
    y_test = pd.read_csv(os.path.join(data_path, 'test_labels.csv'))

    x_train = pd.read_csv(os.path.join(data_path, 'train_new.csv'))
    x_test = pd.read_csv(os.path.join(data_path, 'test_new.csv'))

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.values.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.values.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary(lablel转化为2进制)
    y_true = np.argmax(y_test, axis=1)

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension
        x_train = x_train.values.reshape((x_train.shape[0], x_train.shape[1]))
        x_test = x_test.values.reshape((x_test.shape[0], x_test.shape[1]))


    print(x_train.shape)


    # train
    parameters = {'n_neighbors': [2, 4, 8]}
    clf = GridSearchCV(KNeighborsClassifier(metric=DTW), parameters, cv=3, verbose=1)
    clf.fit(x_train, y_train)

    # evaluate
    y_pred = clf.predict(x_test)
    print(classification_report(y_test, y_pred))



if __name__ == '__main__':
    data_path= r'/code/tsc/dataset/11lei_ori_video_experiment/data_train'
    create_classifier(data_path)


