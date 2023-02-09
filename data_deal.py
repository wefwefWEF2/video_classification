# -*- coding: utf-8 -*-
import os
from imp import reload

import pandas as pd
import re
import os
import glob
import pathlib
import pandas as pd
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys



#计算文件帧数
def readline_count(file_name):
    return len(open(file_name).readlines())

#排除行数不够的文件(frame不够) 并给所有原始文件添加label
def file_select(path_ori,path_new,frame_num):
    os.chdir(path_ori)
    files = os.listdir(path_ori )
    all_files = glob.glob(path_ori + '/' + '*.csv')
    all_files = sorted([pathlib.Path(i) for i in all_files])
    for file in files:
        if file.endswith(".csv"):
            if readline_count(file) >= frame_num:
                file_df= pd.concat([pd.read_csv(file,header=None).assign(New=os.path.basename(file).split('.')[0])  # 增加列为文件名字（添加label）
                               ])
                file_df.columns = ['name', 'time', 'size','label']  #列名字

                file_df['label'] = file_df['label'].apply(lambda x: re.sub(r'^([a-zA-Z]+).*', r'\1', x)) #只保留label字母 去掉数字等其他参数
                file_df.to_csv(os.path.join(path_new, file))  # set the output file location and name at the sametime
                #print("succ")


#path_new内所有文件批量按序号重命名
def rename(path_new):
    i = 0
    files = os.listdir(path_new)   #该文件夹下所有的文件（包括文件夹）
    for file in files:   #遍历所有文件
        i = i + 1
        Olddir = os.path.join(path_new, file)    #原来的文件路径
        if os.path.isdir(Olddir):       #如果是文件夹则跳过
                continue
        filetype = '.csv'        #文件扩展名
        Newdir = os.path.join(path_new, str(i) + filetype)   #新的文件路径
        os.rename(Olddir, Newdir)    #重命名
    return True

#提取序号文件对应的label 生成对应的target文件
def target_deal(path_new,path_train):
    out_file1 = os.path.join(path_new, 'target.csv')# 生成target路径
    out_file2 = os.path.join(path_train, 'target.csv')  # 生成target路径
    all_files = glob.glob(path_new + '/'+'*.csv')
    all_files = sorted([pathlib.Path(i) for i in all_files])
    for fn in all_files:
        temp = pd.read_csv(fn, usecols=['label'], nrows=1, skiprows=0)  # usecols保留列，nrows：需要读取的行数， skiprows=0 跳过的行数
        temp['filename'] = fn.stem  # 新增列 为目录下所有文件名
        temp.rename(columns={'filename': '#sequence_ID', 'label': 'class_label'}, inplace=True)  # 列重命名
        temp = temp[['#sequence_ID', 'class_label']]  # 列交换顺序
        temp.to_csv(out_file1, mode='a', index=False, header=not os.path.isfile(out_file1))
        temp.to_csv(out_file2, mode='a', index=False, header=not os.path.isfile(out_file2))



# 把所有文件合并(改变选择的合并帧数frame_num)  并分成train validation set
def data_combine(path_new,path_train,frame_num):
    #save
    X_train_path= os.path.join(path_train,'train.csv')
    X_test_path = os.path.join(path_train, 'test.csv')
    train_labels_path=os.path.join(path_train,'train_labels.csv')
    test_labels_path = os.path.join(path_train, 'test_labels.csv')


    data_path = ' '.join(sys.argv[1:])  # enter dataset path from command line
    read_path = path_new

    # 设置文件label
    target_df = pd.read_csv(os.path.join(data_path, read_path, r'target.csv'))
    labels = target_df['class_label']

    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    sequence_ids = target_df['#sequence_ID']

    train_ids, test_ids, train_labels, test_labels = train_test_split(sequence_ids, labels, test_size=0.2)

    for sequence in train_ids:
        # 读取所有名字为序号sequence的文件
        df = pd.read_csv(os.path.join(data_path, read_path, f'{sequence}.csv'), nrows=frame_num)  # nrows改变train读取帧数
        # 插入第一列sequence
        df.insert(0, 'sequence', sequence)
        df['step'] = np.arange(df.shape[0])
        X_train = pd.concat([X_train, df])
    for sequence in test_ids:
        df = pd.read_csv(os.path.join(data_path, read_path, f'{sequence}.csv'), nrows=frame_num)  # nrows改变test读取帧数
        df.insert(0, 'sequence', sequence)
        df['step'] = np.arange(df.shape[0])
        X_test = pd.concat([X_test, df])

    X_train.head()
    X_test.head()
    X_train = X_train.drop(columns=["Unnamed: 0", "name", "label"])
    X_test = X_test.drop(columns=["Unnamed: 0", "name", "label"])
    X_train.to_csv(X_train_path,index=False)
    X_test.to_csv(X_test_path, index=False)
    train_labels.to_csv(train_labels_path, index=False)
    test_labels.to_csv(test_labels_path, index=False)




#将所有合并数据处理为横列排列 输入训练的模式
##数据集横向处理  第一列为id 横坐标为随时间变化的size

def data_combine_train(path_train):
    ##训练集测试集读取
    train_ori_path = os.path.join(path_train, 'train.csv')
    test_ori_path = os.path.join(path_train, 'test.csv')
    train_new_sequence_path = os.path.join(path_train, 'train_new_sequence.csv') #带有sequence的文件  用于与label检查
    test_new_sequence_path = os.path.join(path_train, 'test_new_sequence.csv') #带有sequence的文件  用于与label检查
    train_new_path = os.path.join(path_train, 'train_new.csv')#训练集
    test_new_path = os.path.join(path_train, 'test_new.csv')#验证集
    train_label = os.path.join(path_train, 'train_labels.csv')
    test_label = os.path.join(path_train, 'test_labels.csv')
    data_combine = os.path.join(path_train, 'test_combine.csv')#训练集验证集文件合并 用于后续检查
    label_combine = os.path.join(path_train, 'label_combine.csv')
    train_ori = pd.read_csv(train_ori_path)
    test_ori = pd.read_csv(test_ori_path)


    # train_csv处理
    df_new = train_ori.drop(columns=[ "time", "step"])  # 去掉不用列
    v = df_new.melt(id_vars=['sequence'])
    v['variable'] += v.groupby(['sequence', 'variable']).cumcount().astype(str)
    res = v.pivot_table(index=['sequence'], columns='variable', values='value',sort=False)  # sort=False  注意不改变索引顺序 以和label对应
    c = res.columns.str.extract(r'(\d+)')[0].values.astype(int)  # 取值 'variable' 变量size
    np.argsort(c)
    # print(c)
    res.iloc[:, np.argsort(c)].to_csv(
        train_new_sequence_path)  # size0   size1   size2  ...  size2997  size2998  size2999 排序并保存
    res.iloc[:, np.argsort(c)].to_csv(train_new_path, index=False)  # index=False 不保存索引，删除序列号用于检测
    print(res.iloc[:, np.argsort(c)])

    # test_csv处理
    df_new1 = test_ori.drop(columns=["time", "step"])  # 去掉不用列
    v = df_new1.melt(id_vars=['sequence'])
    v['variable'] += v.groupby(['sequence', 'variable']).cumcount().astype(str)
    res = v.pivot_table(index=['sequence'], columns='variable', values='value',
                        sort=False)  # sort=False  注意不改变索引顺序 以和label对应
    c = res.columns.str.extract(r'(\d+)')[0].values.astype(int)  # 取值 'variable' 变量size
    np.argsort(c)
    # print(c)
    res.iloc[:, np.argsort(c)].to_csv(
        test_new_sequence_path)  # size0   size1   size2  ...  size2997  size2998  size2999 排序并保存
    res.iloc[:, np.argsort(c)].to_csv(test_new_path, index=False)
    print(res.iloc[:, np.argsort(c)])

    ####train test数据合并 测试集使用，训练集不使用以下文件
    df1 = pd.read_csv(test_new_path)
    df2 = pd.read_csv(train_new_path)
    df = pd.concat([df1, df2])  # 合并
    # df.to_csv(data_combine,encoding = 'utf-8',index=False)
    df.to_csv(data_combine, encoding='utf-8', index=['sequence'])
    print(df)
    ####label数据合并 测试集使用，训练集不使用以下文件
    df1 = pd.read_csv(test_label)
    df2 = pd.read_csv(train_label)
    df = pd.concat([df1, df2])  # 合并
    # df.to_csv(label_combine,encoding = 'utf-8',index=False)
    df.to_csv(label_combine, encoding='utf-8', index=['sequence'])


if __name__ == '__main__':
    #path_ori 原始文件
    #path_new 筛选完帧数的文件
    #path_train选择合并帧数文件
    path_ori = r'/code/tsc/dataset/data/data_ori'
    path_new = r'/code/tsc/dataset/data/data_new'
    path_train=r'/code/tsc/dataset/data/data_train'

    file_select(path_ori, path_new,3000)
    rename(path_new)
    target_deal(path_new,path_train)
    data_combine(path_new, path_train, 3000)
    data_combine_train(path_train)


