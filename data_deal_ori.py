import pandas as pd
import os
import pandas as pd
import numpy as np
#四
##  数据集横向处理  第一列为id 横坐标为随时间变化的size
train_ori = pd.read_csv('/home/dyn/paper_experiment/tsc/dataset/frame_test/train.csv')
test_ori = pd.read_csv('/home/dyn/paper_experiment/tsc/dataset/frame_test/test.csv')


#
# df1 = train_ori.assign(cid = train_ori.groupby(['sequence']).cumcount()).set_index(['sequence','size']).unstack(-1).sort_index()
# df1.columns = [f'{x}{y}' for x,y in df1.columns]
# df1 = df1.reset_index()

#train_csv处理
df_new = train_ori.drop(columns=["Unnamed: 0","time","step"])#去掉不用列
v =df_new.melt(id_vars=['sequence'])
v['variable']+= v.groupby(['sequence','variable']).cumcount().astype(str)

res = v.pivot_table(index=['sequence'], columns='variable', values='value',sort=False) #sort=False  注意不改变索引顺序 以和label对应

c = res.columns.str.extract(r'(\d+)')[0].values.astype(int)  #取值 'variable' 变量size

np.argsort(c)
# print(c)
res.iloc[:,np.argsort(c)].to_csv('/home/dyn/paper_experiment/tsc/dataset/frame_test/train_new.csv') # size0   size1   size2  ...  size2997  size2998  size2999 排序并保存


print(res.iloc[:,np.argsort(c)])





#test_csv处理
df_new1 = test_ori.drop(columns=["Unnamed: 0","time","step"])#去掉不用列
v =df_new1.melt(id_vars=['sequence'])
v['variable']+= v.groupby(['sequence','variable']).cumcount().astype(str)

res = v.pivot_table(index=['sequence'], columns='variable', values='value',sort=False) #sort=False  注意不改变索引顺序 以和label对应

c = res.columns.str.extract(r'(\d+)')[0].values.astype(int)  #取值 'variable' 变量size

np.argsort(c)
# print(c)
res.iloc[:,np.argsort(c)].to_csv('/home/dyn/paper_experiment/tsc/dataset/frame_test/test_new.csv') # size0   size1   size2  ...  size2997  size2998  size2999 排序并保存


print(res.iloc[:,np.argsort(c)])