import numpy as np
import pandas as pd
import sklearn
import random
import tensorflow as tf
from tensorflow import keras
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
import os
import timeit
import tensorflow.keras as keras
import numpy as np
import time
import pandas as pd
import matplotlib
from utils.utils import save_test_duration
matplotlib.use('agg')
from utils.utils import calculate_metrics
import sklearn

from utils.utils import read_all_datasets


def setup_seed(seed):
    random.seed(seed)  # 为python设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    tf.random.set_seed(seed)  # tf cpu fix seed
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  # tf gpu fix seed, please `pip install tensorflow-determinism` first


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

#use time_size as input
def fit_classifier(data_path):
    #x_train, x_test, y_train, y_test = train_test_split(train, train_labels)
    
    y_train = pd.read_csv(os.path.join(data_path,'train_labels.csv'))
    y_test = pd.read_csv(os.path.join(data_path,'test_labels.csv'))

    x_train = pd.read_csv(os.path.join(data_path,'train_new.csv'))
    x_test = pd.read_csv(os.path.join(data_path,'test_new.csv'))
    
    x_train = x_train.iloc[:, :3000] 
    y_train = y_train[:] 
    x_test = x_test.iloc[:, :3000] 
    y_test = y_test[:] 


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
    output_directory="/code/tsc/result/deepfake/++/qp23/"
    classifier = create_classifier(classifier_name,input_shape, nb_classes,output_directory)
    classifier.fit(x_train, y_train, x_test, y_test, y_true)



if __name__ == '__main__':
    log = open("/code/tsc/result/30f_comparewith_dtw/resnet.txt",mode="a",encoding="utf-8")
    start = timeit.default_timer()
    data_path= r'/code/tsc/dataset/deep_fake/++/face_swap/qp23/train/data_train'
    setup_seed(100)
    fit_classifier(data_path)
    end = timeit.default_timer()
    print('Running time: %s Seconds' % (end - start),file = log)
    log.close()
