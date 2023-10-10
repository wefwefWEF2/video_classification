import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### Set your path to the folder containing the .csv files
from matplotlib.ticker import MultipleLocator

PATH =r""

### Fetch all files in path
fileNames = os.listdir(PATH)

### Filter file name list for files ending with .csv
fileNames = [file for file in fileNames if '.csv' in file]




### Loop over all files
for file in fileNames:

    newFileName = file.split(".")[0]
    savePath=r'C:\Users\18686\Desktop\ceshiji\picture\\'
    ### Read .csv file and append to list
    df = pd.read_csv(PATH + file, skiprows=0)
    # print(df['num'])
    plt.show()

    ### Generate the plot
    plt.figure(figsize=(20,8),dpi=80)#画布大小
    plt.xlabel('frame',fontsize = 14)  # x轴标题
    plt.ylabel('frame size in bits',fontsize = 14)  # Y轴标题
    ax = plt.gca()#间隔
    ax.xaxis.set_major_locator(MultipleLocator(1000) )
    ax.yaxis.set_major_locator(MultipleLocator(100000))



    plt.plot(df.iloc[:, 1],df.iloc[:, 2])# 读取第二列 第三列
    plt.savefig(fname=savePath + newFileName + ".png")
