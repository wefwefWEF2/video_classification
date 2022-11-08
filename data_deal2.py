
import os
import pandas as pd
import numpy as np

##  数据集横向处理  第一列为id 横坐标为随时间变化的size


##处理数据读取和保存路径
path_ori=r'F:\qinghua_intership\project\tsc_new\dataset\frame_test'


##训练集测试集读取
train_ori_path= os.path.join(path_ori,'train.csv')
test_ori_path = os.path.join(path_ori,'test.csv')
train_new_sequence_path= os.path.join(path_ori,'train_new_sequence.csv')
test_new_sequence_path = os.path.join(path_ori,'test_new_sequence.csv')
train_new_path= os.path.join(path_ori,'train_new.csv')
test_new_path = os.path.join(path_ori,'test_new.csv')
train_label= os.path.join(path_ori,'train_labels.csv')
test_label= os.path.join(path_ori,'test_labels.csv')

data_combine=os.path.join(path_ori,'test_combine.csv')
label_combine=os.path.join(path_ori,'label_combine.csv')

train_ori = pd.read_csv(train_ori_path)
test_ori = pd.read_csv(test_ori_path)
#train_ori = pd.read_csv(r'F:\qinghua_intership\project\tsc_new\dataset\frame=3000(8)(250)\train.csv')
#test_ori = pd.read_csv(r'F:\qinghua_intership\project\tsc_new\dataset\frame=3000(8)(250)\test.csv')


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
res.iloc[:,np.argsort(c)].to_csv(train_new_sequence_path) # size0   size1   size2  ...  size2997  size2998  size2999 排序并保存
res.iloc[:,np.argsort(c)].to_csv(train_new_path,index=False)#index=False 不保存索引，删除序列号用于检测

print(res.iloc[:,np.argsort(c)])





#test_csv处理
df_new1 = test_ori.drop(columns=["Unnamed: 0","time","step"])#去掉不用列
v =df_new1.melt(id_vars=['sequence'])
v['variable']+= v.groupby(['sequence','variable']).cumcount().astype(str)
res = v.pivot_table(index=['sequence'], columns='variable', values='value',sort=False) #sort=False  注意不改变索引顺序 以和label对应
c = res.columns.str.extract(r'(\d+)')[0].values.astype(int)  #取值 'variable' 变量size
np.argsort(c)
# print(c)
res.iloc[:,np.argsort(c)].to_csv(test_new_sequence_path) # size0   size1   size2  ...  size2997  size2998  size2999 排序并保存
res.iloc[:,np.argsort(c)].to_csv(test_new_path,index=False)
print(res.iloc[:,np.argsort(c)])

####train test数据合并 测试集使用，训练集不使用以下文件
df1 = pd.read_csv(test_new_path)
df2 = pd.read_csv(train_new_path)
df = pd.concat([df1,df2])#合并
df.to_csv(data_combine,encoding = 'utf-8')
print(df)
####label数据合并 测试集使用，训练集不使用以下文件
df1 = pd.read_csv(test_label)
df2 = pd.read_csv(train_label)
df = pd.concat([df1,df2])#合并
#df.drop_duplicates()  #数据去重
df.to_csv(label_combine,encoding = 'utf-8')