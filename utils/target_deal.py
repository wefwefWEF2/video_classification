import os
import glob
import pathlib

import pandas as pd

path = r'/home/dyn/paper_experiment/video_classification/dataset/data_test/data_new//'   #数据集路径
out_file = r'/home/dyn/paper_experiment/video_classification/dataset/data_test/data_new/target.csv' #生成target路径
all_files = glob.glob(path + '*.csv')
all_files = sorted([pathlib.Path(i) for i in all_files])



for fn in all_files:
    temp = pd.read_csv(fn, usecols=['label'],nrows=1, skiprows=0)#usecols保留列，nrows：需要读取的行数， skiprows=0 跳过的行数
    temp['filename'] = fn.stem   #新增列 为目录下所有文件名
   
    temp.rename(columns={'filename': '#sequence_ID', 'label': 'class_label'}, inplace=True) #列重命名
    temp=temp[['#sequence_ID', 'class_label']] #列交换顺序
    temp.to_csv(out_file, mode='a', index=False, header=not os.path.isfile(out_file))

