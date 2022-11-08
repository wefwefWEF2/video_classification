import pandas as pd
import os
import pandas as pd


train_ori = pd.read_csv(r'train.csv')
test_ori = pd.read_csv(r'F:\qinghua_intership\project\video_classification1\test.csv')


df1 = train_ori.assign(cid = train_ori.groupby(['sequence']).cumcount()).set_index(['size']).unstack(-1).sort_index(1,1)

df1.columns = [f'{x}{y}' for x,y in df1.columns]

df1 = df1.reset_index()