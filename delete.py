#删除训练集测试集内容相同文件
import os
import pandas as pd


##读取data_new文件
data_new_path =os.path.join(r'path','data_new')    #数据集路径
test_new_path =os.path.join(r'path','data_test')  #测试集路径



# all_train_files = glob.glob(data_new_path + '*.csv')
# all_train_files = sorted([pathlib.Path(i) for i in all_train_files])
# all_test_files = glob.glob(test_new_path + '*.csv')
# all_test_files = sorted([pathlib.Path(i) for i in all_train_files])
# file1=pd.read_csv(r'F:\qinghua_intership\project\tsc_new\dataset\data_new\5.csv')
# file2=pd.read_csv(r'F:\qinghua_intership\project\tsc_new\dataset\data_new\5.csv')
#result = filecmp.cmp(file1, file2)


def compare_file(file1,file2):
    column1=file1['size'].values.tolist()
    #print(len(column1))
    column2 =file2['size'].values.tolist()
    #print(len(column2))
    #修改比较前多少个值
    column1_20 = column1[0:5]
    column2_20 = column2[0:5]
    #判断相同文件
    if (len(column1)==len(column2)) & (column1_20==column2_20):
        #print(file1.name)
        #print(file2.name)
        return True

def compare_folder(data_new_path, test_new_path):
    files1 = os.listdir(data_new_path)
    files2 = os.listdir(test_new_path)
    for file1 in files1:
        f1=pd.read_csv(os.path.join(data_new_path,file1))
        #f1 = open(data_new_path + "/" + file1,encoding = 'utf-8').read()
        for file2 in files2:
            f2 = pd.read_csv(os.path.join(test_new_path, file2))
            if compare_file(f1, f2):
                print('删除训练文件'+file1.title()+'相同测试集文件为:'+file2.title())
                #删除data_new里的文件
                #os.remove(data_new_path+"/"+file1)


if __name__ == '__main__':
    compare_folder(data_new_path, test_new_path)





