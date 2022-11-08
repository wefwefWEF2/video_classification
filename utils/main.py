import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from models import cnn_tsc
from utils.utils import generate_results_csv
from utils.utils import create_directory
from utils.utils import read_dataset
from utils.utils import transform_mts_to_ucr_format
from utils.utils import visualize_filter
from utils.utils import viz_for_survey_paper
from utils.utils import viz_cam
import os
import numpy as np
import sys
import sklearn
import utils

from utils.utils import read_all_datasets


# # use train_features as input
# train_features = pd.read_csv('/home/dyn/paper_experiment/tsc/dataset/frame=3000(9)/train_features.csv')
# test_features = pd.read_csv('/home/dyn/paper_experiment/tsc/dataset/frame=3000(9)/test_features.csv')
#
# # use time-size as input
# #修改标签！！！
# train_labels = pd.read_csv(r'/home/dyn/paper_experiment/tsc/dataset/frame=3000(9)/train_labels.csv')
# test_labels = pd.read_csv(r'/home/dyn/paper_experiment/tsc/dataset/frame=3000(9)/test_labels.csv')
#
# train = pd.read_csv('/home/dyn/paper_experiment/tsc/dataset/frame=3000(9)/train_new.csv')
# test = pd.read_csv('/home/dyn/paper_experiment/tsc/dataset/frame=3000(9)/test_new.csv')


# os.path.join()
def fit_classifier():
    y_train = pd.read_csv(r'/home/dyn/paper_experiment/tsc/dataset/frame=1000(9)/train_labels.csv')
    y_test = pd.read_csv(r'/home/dyn/paper_experiment/tsc/dataset/frame=1000(9)/test_labels.csv')

    x_train = pd.read_csv('/home/dyn/paper_experiment/tsc/dataset/frame=1000(9)/train_features.csv')
    x_test = pd.read_csv('/home/dyn/paper_experiment/tsc/dataset/frame=1000(9)/test_features.csv')

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.values.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.values.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    y_true = np.argmax(y_test, axis=1)

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 增加一维说明 一个时间步长有1个变量  9维为9个变量
        x_train = x_train.values.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.values.reshape((x_test.shape[0], x_test.shape[1], 1))

    input_shape = x_train.shape[1:]
    classifier_name = "resnet"
    output_directory=""
    classifier = create_classifier(classifier_name,input_shape, nb_classes,output_directory)
    classifier.fit(x_train, y_train, x_test, y_test, y_true)

def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=False):
    if classifier_name == 'fcn':
        from models import fcn
        return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mlp':
        from models import mlp
        return mlp.Classifier_MLP(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'resnet':
        from models import resnet
        return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcnn':
        from models import mcnn
        return mcnn.Classifier_MCNN(output_directory, verbose)
    if classifier_name == 'tlenet':
        from models import tlenet
        return tlenet.Classifier_TLENET(output_directory, verbose)
    if classifier_name == 'twiesn':
        from models import twiesn
        return twiesn.Classifier_TWIESN(output_directory, verbose)
    if classifier_name == 'encoder':
        from models import encoder
        return encoder.Classifier_ENCODER(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcdcnn':
        from models import mcdcnn
        return mcdcnn.Classifier_MCDCNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'cnn_tsc':  # Time-CNN
        from models import cnn_tsc
        return cnn_tsc.Classifier_CNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'inception':
        from models import inception
        return inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose)





# change function name
#use time_size as input
def fit_classifier2():
    #x_train, x_test, y_train, y_test = train_test_split(train, train_labels)
    y_train = pd.read_csv(r'/home/dyn/paper_experiment/tsc/dataset/frame=3000(6)(250)/train_labels.csv')
    y_test = pd.read_csv(r'/home/dyn/paper_experiment/tsc/dataset/frame=3000(6)(250)/test_labels.csv')

    x_train = pd.read_csv('/home/dyn/paper_experiment/tsc/dataset/frame=3000(6)(250)/train_new.csv')
    x_test = pd.read_csv('/home/dyn/paper_experiment/tsc/dataset/frame=3000(6)(250)/test_new.csv')

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
        x_train = x_train.values.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.values.reshape((x_test.shape[0], x_test.shape[1], 1))

    input_shape = x_train.shape[1:]
    classifier_name = "resnet"
    output_directory=""
    classifier = create_classifier(classifier_name,input_shape, nb_classes,output_directory)
    classifier.fit(x_train, y_train, x_test, y_test, y_true)



#fit_classifier()
fit_classifier2()