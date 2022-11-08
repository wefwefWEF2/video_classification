import os
import sys

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

data_path = ' '.join(sys.argv[1:]) # enter dataset path from command line

#设置文件label
target_df = pd.read_csv(os.path.join(data_path, 'dataset/data_test/data_new', r'target.csv'))
labels = target_df['class_label']
#labels.rename('class_label', inplace=True) # remove the leading whitespace

X_train = pd.DataFrame()
X_test = pd.DataFrame()

sequence_ids = target_df['#sequence_ID']

train_ids, test_ids, train_labels, test_labels = train_test_split(sequence_ids, labels, test_size=0.99)

for sequence in train_ids:
    #更改全部输入文件名字 
    df = pd.read_csv(os.path.join(data_path, 'dataset/data_test/data_new', f'{sequence}.csv'),nrows=3000) #nrows改变train读取帧数
    #插入第一列sequence
    #df.insert(0, 'sequence', sequence-1)？？？
    df.insert(0, 'sequence', sequence)
    df['step'] = np.arange(df.shape[0])
    X_train = pd.concat([X_train, df])
    # 需要过滤未命名列
    #df = df.loc[:, ~df.columns.str.contains('Unnamed: 5')]
    print(df)

for sequence in test_ids:
    df = pd.read_csv(os.path.join(data_path, 'dataset/data_test/data_new', f'{sequence}.csv'),nrows=3000) #nrows改变test读取帧数
    #可能与label对应不上？？？？
    #df.insert(0, 'sequence', sequence-1)
    df.insert(0, 'sequence', sequence)
    df['step'] = np.arange(df.shape[0])
    X_test = pd.concat([X_test, df])
    print(df)



##文件保存

path_to_save=r'/home/dyn/paper_experiment/tsc/dataset/frame=3000(8)(250)'
##保存路径及名字
X_train= os.path.join(path_to_save,'train.csv')
X_test= os.path.join(path_to_save,'test.csv')
train_labels= os.path.join(path_to_save,'train_labels.csv')
test_labels = os.path.join(path_to_save,'test_labels.csv')

X_train.to_csv(X_train, index=False)
X_test.to_csv(X_test, index=False)
train_labels.to_csv(train_labels, index=False)
test_labels.to_csv(test_labels, index=False)


###删除未命名列
#修改train.csv
df1 = pd.read_csv(X_train)
df1.head()
# 2.删除指定列
df_new = df1.drop(columns=["Unnamed: 0","name","label"])
# 3.保存删除后的数据内容
df_new.to_csv(X_train)


#修改test.csv
df2 = pd.read_csv(X_test)
df2.head()
# 2.删除指定列
df_new = df2.drop(columns=["Unnamed: 0","name","label"])
# 3.保存删除后的数据内容
df_new.to_csv(X_test)
