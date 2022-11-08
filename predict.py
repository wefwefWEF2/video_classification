
import numpy as np
# resnet model
# when tuning start with learning rate->mini_batch_size ->
# momentum-> #hidden_units -> # learning_rate_decay -> #layers
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time
import pandas as pd
import matplotlib
from utils.utils import save_test_duration
matplotlib.use('agg')
from utils.utils import calculate_metrics
import sklearn


#查看预测结果
#导入npy文件路径位置
# y_pred = np.load(r'F:\qinghua_intership\project\paper_experiment\tsc\result\y_pred.npy')
#
# # convert the predicted from binary to integer
# y_pred = np.argmax(y_pred, axis=1)
# print(y_pred)




def predict(x_test, y_true, return_df_metrics=True):
    start_time = time.time()
    model_path = r'/home/dyn/paper_experiment/tsc/best_model.hdf5'
    model = keras.models.load_model(model_path)
    y_pred = model.predict(x_test)
    print( y_pred )
    if return_df_metrics:
        y_pred = np.argmax(y_pred, axis=1)
        df_metrics = calculate_metrics(y_true, y_pred, 0.0)
        print(df_metrics)
        return df_metrics
    else:
        test_duration = time.time() - start_time
        save_test_duration('test_duration.csv', test_duration)
        return y_pred


if __name__ == '__main__':
    # x_train, x_test, y_train, y_test = train_test_split(train, train_labels)
    y_train = pd.read_csv(
        r'/home/dyn/paper_experiment/tsc/dataset/frame_test/train_labels.csv')
    y_test = pd.read_csv(
        r'/home/dyn/paper_experiment/tsc/dataset/frame_test/test_labels.csv')

    x_train = pd.read_csv(
        r'/home/dyn/paper_experiment/tsc/dataset/frame_test/train_new.csv')
    x_test = pd.read_csv(r'/home/dyn/paper_experiment/tsc/dataset/frame_test/test_new.csv')

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.values.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.values.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary(lablel转化为2进制)
    y_true = np.argmax(y_test, axis=1)
    print(y_true)

    predict(x_test = pd.read_csv(r'/home/dyn/paper_experiment/tsc/dataset/frame_test/test_new.csv'),
            y_true = np.argmax(y_test, axis=1),
            return_df_metrics=True)