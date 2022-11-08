
import os
import pandas as pd
import re
#添加label  新生成文件输出到data_new
path = r'/home/dyn/paper_experiment/video_classification/dataset/data_test/data_ori'
path2 = r'/home/dyn/paper_experiment/video_classification/dataset/data_test/data_new'
os.chdir(path)

files = os.listdir(path)

# (os.path.join(path2,r'green1.csv'))

#计算文件帧数
def readline_count(file_name):
    return len(open(file_name).readlines())


for file in files:
    #排除行数不够的文件(frame不够)
    if readline_count(file) >3000:

        file_df= pd.concat([pd.read_csv(file,header=None).assign(New=os.path.basename(file).split('.')[0])  # 增加列为文件名字（添加label）
                       ])
        print(file_df)

        file_df.columns = ['name', 'time', 'size','label']  #列名字

        file_df['label'] = file_df['label'].apply(lambda x: re.sub(r'^([a-zA-Z]+).*', r'\1', x)) #只保留label字母 去掉数字等其他参数
        file_df.to_csv(os.path.join(path2, file))  # set the output file location and name at the sametime
        print("succ")



