
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






def predict(x_test, y_true, return_df_metrics=True):
    start_time = time.time()

    model_path = os.path.join(r'/code/tsc/result/1000k/40_60e=150_b=8/frame=3000','best_model.hdf5')
    model = keras.models.load_model(model_path)
    y_pred = model.predict(x_test)
    y_pred_trans = np.argmax(y_pred, axis=1)#预测值转为二进制
    if return_df_metrics:
        y_pred = np.argmax(y_pred, axis=1)
        df_metrics = calculate_metrics(y_true, y_pred, 0.0)
        print(df_metrics)
        print("################Predict results###########")
        print(y_pred_trans)
        return df_metrics
    else:
        test_duration = time.time() - start_time
        save_test_duration('test_duration.csv', test_duration)
        return y_pred


if __name__ == '__main__':
    start = timeit.default_timer()
    y_test = pd.read_csv(os.path.join(r'/code/tsc/dataset/1000k_3000f_experiment','test_labels.csv'))

    x_test = pd.read_csv(os.path.join(r'/code/tsc/dataset/1000k_3000f_experiment','test_new.csv'))

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.array((y_test)).reshape(-1, 1))
    y_test = enc.transform(y_test.values.reshape(-1, 1)).toarray()
    # save orignal y because later we will use binary
    y_true = np.argmax(y_test, axis=1)


    predict(x_test = pd.read_csv(r'/code/tsc/dataset/1000k_3000f_experiment/test_new.csv'),y_true=y_true,return_df_metrics=True)
    print("################True results###########")
    print(y_true)

    end = timeit.default_timer()
    print('Running time: %s Seconds' % (end - start))
