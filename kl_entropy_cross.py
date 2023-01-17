#!/usr/bin/env python
# coding=utf-8
# author: chicho
# running: python trees.py
# filename : trees.py
import os
from itertools import chain
from math import log
import scipy.stats
import numpy as np
import pandas as pd
from scipy.stats import entropy
def interval_statistics(data, intervals):
    if len(data) == 0:
        return
    for num in data:
        for interval in intervals:
            lr = tuple(interval.split('~'))
            left, right = float(lr[0]), float(lr[1])
            if left <= num <= right:
                intervals[interval] += 1
                break

    for key, value in intervals.items():
        print("%10s" % key, end='')  # 借助 end=''可以不换行
        print("%10s" % value, end='')  # "%10s" 右对齐
        print('%16s' % '{:.3%}'.format(value * 1.0 / len(data)))


def calcShannonEnt(dataSet):
    countDataSet = len(dataSet)
    print(countDataSet)
    labelCounts={}
    for featVec in dataSet:
        currentLabel=featVec
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    print("统计为")
    print( labelCounts)

    shannonEnt = 0.0
    Shannon2=0.0
    prob_sum= 0.0
    Shannon1=0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/countDataSet
        shannonEnt =-prob * log(prob,2)
        print(float(shannonEnt))
        Shannon2 += shannonEnt
        prob_sum += prob
        print("统计为")
        # print(prob_sum)
        #print(Shannon)
    return Shannon2

#计算间隔为i的entropy
def entropy_interval(data, intervals):
    if len(data) == 0:
        return
    for num in data:
        for interval in intervals:
            lr = tuple(interval.split('~'))
            left, right = float(lr[0]), float(lr[1])
            if left <= num <= right:
                intervals[interval] += 1
                break

    countDataSet = len(data)
    print(countDataSet)
    shannonEnt = 0.0
    Shannon2 = 0.0
    prob_sum = 0.0
    for key, value in intervals.items():
        prob = float(value / countDataSet)
        shannonEnt = -prob * log(prob, 2)
        print(float(shannonEnt))
        Shannon2 += shannonEnt
        prob_sum += prob
        print("统计为")
        print(prob_sum)
        print(Shannon2)
    return Shannon2

#将数据展成一行
def trans_to_calculate(data):
    # test_combine1 = pd.read_csv(data_path)

    test = np.array(data)
    # print(test.shape)
    # print(label.shape)
    test = test.tolist()
    c = list(chain(*test))
    dataSet = [i for item in test for i in item]
    dataSet = np.array(dataSet).transpose()
    # 传入c为n行一列的数组形式
    c = np.array(c)
    print(c.shape)
    return c




#计算间隔为interval,并且去分间隔统计的kl散度 cross entropy entropy
def kl_interval(data1,data2,intervals1,intervals2):
    shannonEnt = 0.0
    KL = 0.0
    prob_sum = 0.0
    cross_entropy = 0.0

    data1= trans_to_calculate(data1)
    data2 = trans_to_calculate(data2)
#统计data1
    for num in data1:
        for interval1 in intervals1:
            lr = tuple(interval1.split('~'))
            left, right = float(lr[0]), float(lr[1])
            if left <= num <= right:
                intervals1[interval1] += 1
                break

    countDataSet1 = len(data1)
    countDataSet2 = len(data2)
    #打印查看数据长度信息
    # print(countDataSet1)
    prob_sum = 0.0
#统计data2
    for num in data2:
        for interval2 in intervals2:
            lr = tuple(interval2.split('~'))
            left, right = float(lr[0]), float(lr[1])
            if left <= num <= right:
                intervals2[interval2] += 1
                break

    countDataSet2 = len(data2)

 # 分区间计算kl entropy

    # for value in intervals1.items() and intervals2.items():
    #for i in range(0,10000000000000000):

    value1=list(intervals1.values())
    value1 = np.array(value1)
    value1=value1/countDataSet1
    value2 = list(intervals2.values())
    value2 = np.array(value2)
    value2 = value2 / countDataSet2
    # print(value2)
    prob2 = value2
    prob1 = value1
    for i in range(0, 200):
        if prob2[i] == 0 or prob1[i] == 0:
            break
        else:

            #KL += prob1[i] * np.log(prob1[i] /prob2[i])
            KL += prob1[i] *(log(prob1[i], 2)-log(prob2[i], 2))
            cross_entropy += -prob1[i] * log(prob2[i], 2)
            shannonEnt += -prob1[i]* log(prob1[i], 2)
            prob_sum+=prob2[i]
            kl_check=cross_entropy-shannonEnt
    # print("0-200qujian")
    # print("kl")
    # # print(prob_sum)
    # print(KL)
    # print("entropy")
    # print(shannonEnt)
    #print("cross-entropy")
    #print(cross_entropy)


    for i in range(200, 2000000000000):
        if prob2[i] == 0 or prob1[i] == 0:
            break
        else:
            #KL += prob1[i] * np.log(prob1[i] / prob2[i])
            KL += prob1[i] * (log(prob1[i], 2) - log(prob2[i], 2))
            cross_entropy +=-prob1[i]* log(prob2[i], 2)
            shannonEnt += -prob1[i] * log(prob1[i], 2)
            prob_sum += prob2[i]

    # print("200yishangqujian")
    print("kl")
    print(KL)
    print(prob_sum)
    print("entropy")
    print(shannonEnt)
    print("cross-entropy")
    print(cross_entropy)

    return KL












if __name__ == '__main__':

    #区间统计
    start = 0  # 区间左端点
    number_of_interval = 5000# 区间个数
    length =3333 # 区间长度


    # print(intervals)

    # print(data)
    #c=trans_to_calculate(r'cartoon.csv')
    #entropy_interval(c, intervals)
    # interval_statistics(c, intervals)


    #区间为intervals的 kl散度计算aerial
    data1=(r'/code/tsc/entropy/20lei1500k3000fdata/data1')
    data2 = (r'/code/tsc/entropy/20lei1500k3000fdata/data2')
    data1 = (r'/code/tsc/entropy/20lei1500k3000fdata/data1')
    data2 = (r'/code/tsc/entropy/20lei1500k3000fdata/data2')
    files1 = os.listdir(data1)
    files2 = os.listdir(data2)

    for file1 in files1:
        print(os.path.basename(file1))
        f1 = pd.read_csv(os.path.join(data1, file1))
        for file2 in files2:
            f2 = pd.read_csv(os.path.join(data2, file2))
            intervals1 = {'{:.2f}~{:.2f}'.format(length * x + start, length * (x + 1) + start): 0 for x in
                          range(number_of_interval)}  # 生成区间
            intervals2 = {'{:.2f}~{:.2f}'.format(length * x + start, length * (x + 1) + start): 0 for x in
                          range(number_of_interval)}  # 生成区间
            print(os.path.basename(file2))
            #kl_interval(data1, data2, intervals1, intervals2)


            kl_interval(f1, f2, intervals1, intervals2)
























