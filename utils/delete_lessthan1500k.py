import os
import pandas as pd
path=r"F:\qinghua_intership\project\tsc_new\dataset"

##读取码率大小文件
data=os.path.join(r'F:\qinghua_intership\project\tsc_new\dataset\bitrate')

#读取码率
files = os.listdir(data)
f1=pd.read_csv(os.path.join(data,'cartoon.csv'))
column1=f1['码率(kb)'].values.tolist()
# print(column1)
# print(len(column1))

#记录码率不满足要求的列表
mp4_list = []
csv_list = ['.csv']
for i in range(len(column1)):
    if column1[i]<1600:
        # print(column1[i])
        # print(i)
        # print( f1['文件名'][i])
        mp4_list.append(f1['文件名'][i])

print(len(mp4_list))
#print(mp4_list[1])

result = ['{}{}'.format(a, b) for b in csv_list for a in mp4_list]
print(result)









