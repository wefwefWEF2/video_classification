
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time
import pandas as pd
import matplotlib
from tensorboard import summary
from sklearn.ensemble import VotingClassifier
from utils.utils import save_test_duration
matplotlib.use('agg')
from utils.utils import calculate_metrics
import sklearn
import os
import time

model_path=r'/code/tsc/result/800k/40_60e=150_b=8/best_model.hdf5'


#读取模型需要的输入形状
def get_model_size(model_path):
    model = keras.models.load_model(model_path)
    for layer in model.layers:
        input_size=layer.output_shape[0][1]
        print(input_size)
        break
    return  input_size

#移动窗口长度不够模型长度，读取数据末尾填0处理
# def padding_zero(model_path,x_test,window_size):
#
#     if  get_model_size(model_path)<window_size:
#



#满足输入要求的数据直接预测
def predict(x_test, y_true, return_df_metrics=True):
    start_time = time.time()
    model_path = r'/code/tsc/result/1500k/40_60e=150_b=8/frame=1000/best_model.hdf5'
    model = keras.models.load_model(model_path)
    # print("model shape")
    # model.summary()
    y_pred = model.predict(x_test)
    print('##############predirct results#########')
    y_pred_trans = np.argmax(y_pred, axis=1)
    #print(y_pred_trans)
    if return_df_metrics:
        test_duration = time.time() - start_time
        df_metrics = calculate_metrics(y_true, y_pred_trans, test_duration)
        print(df_metrics)
        return df_metrics
    else:
        test_duration = time.time() - start_time
        save_test_duration('test_duration.csv', test_duration)
        return y_pred_trans

#软投票法 输出预测结果较多的值作为结果
# def weighted_predict(predict_results,y_true,split_nums):

#     weight=0
#     for i in





#填充数据符合输入要求,按窗口大小划分样本
def window_predict(source, model_shape,window_size):
    model_shape = get_model_size(model_path)
    source = pd.read_csv(r'/code/tsc/dataset/ceshiji/ceshiji_frame5000/1500_test/frame=5000/test_combine.csv')

    # if source is None or len(source) <= 0:
    #     return source

    source = np.array(source)
    source=np.nan_to_num(source)#nan值(帧数不全)全部换为0
    # padded = pad_sequences(source, padding='post')#末尾补充0

    #分割测试数据
    samples = []
    sw_steps = 1000  # 滑动窗口步长
    simul_data = []
    m=len(source)#行数：数据个数
    n=len(source[0])#列数 每个size个数
    # for i
    for i in range(0, n, sw_steps):
        sample = source[:,i:i + sw_steps]  # 截取所有数据 i 到 i + w(列) 的数据
        samples.append(sample)
    print("共计分割段数量：")
    print(len(samples))#将所有采样点数据，按照不重叠、sw_steps个采样点构成一个样本的方式，划分成n个样本


    predict_results=[]
    for sample in samples:
        sample = np.array(sample)
        y_test = pd.read_csv(r'/code/tsc/dataset/ceshiji/ceshiji_frame5000/1500_test/frame=5000/label_combine.csv')

        enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
        enc.fit(np.array((y_test)).reshape(-1, 1))
        y_test = enc.transform(y_test.values.reshape(-1, 1)).toarray()
        y_true = np.argmax(y_test, axis=1)

        #print('##############True results#########')
        #print(y_true)
        #return_df_metrics=False 仅返回预测结果  用于添加到predict_results
        predict_result=predict(sample,y_true=y_true, return_df_metrics=False)#sample视频分段数据 y_true为视频对应label
        predict_results.append(predict_result)
    return predict_results



# def window_predict_overlap(source, model_shape,window_size):

#移动窗口预测
# def shift_window_predict(source, model_shape,window_size):

#     model_shape=get_model_size(model_path)
#     source = pd.read_csv(r'/code/tsc/dataset/ceshiji/ceshiji_frame3000/1500_test/test_combine.csv')
#     source_label = pd.read_csv(r'/code/tsc/dataset/ceshiji/ceshiji_frame3000/1500_test/label_combine.csv')
#     print(len(source))
#     #判断输入视频数量
#     if source is None or len(source) <= 0:
#         return source
#     #如果范围过大 不需要分段直接预测
#     if window_size >= model_shape:
#         print("window_size should <"+model_shape)
#         # predict(x_test, y_true, return_df_metrics=True)
#     # else:
#     #     predict_window(window_size, data, label, return_df_metrics=True)




# def result_trans():




if __name__ == '__main__':

    window_predict(r'/code/tsc/video_classifiction/dataset/data_1000k/train/data_window_fortrain/test_combine.csv', 1000, 1000)
