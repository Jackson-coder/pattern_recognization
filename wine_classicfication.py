import numpy as np
import csv
from sklearn import preprocessing

#以下两个函数分别求Sw和Sb
def calculate_Sw(data, left, right, length_of_vector, P):
    # mean
    m = np.zeros((length_of_vector,))
    for i in range(left, right):
        m = m + data[i]
    m = m/(right - left)

    total = 0

    for i in range(left, right):
        x_m = data[i]-m
        x_m = np.reshape(x_m, (1, length_of_vector))
        total += np.dot(x_m.T, x_m)

    Sw = 1.0/(right-left) * total * P

    return m, Sw


def calculate_Sb(data, mi, length_of_vector, P):
    m = np.zeros((length_of_vector,))
    for i in data:
        m = m + i
    m = np.array(m/len(data))

    m_m = np.zeros((len(mi), length_of_vector))
    for i in range(len(mi)):
        m_m[i] = np.array(m-mi[i])

    Sb = np.zeros((length_of_vector, length_of_vector))

    for i in range(len(mi)):
        Sb += (np.dot(np.reshape(m_m[i], (1, length_of_vector)
                                 ).T, np.reshape(m_m[i], (1, length_of_vector))))

    Sb *= P

    return m, Sb

# num0:the number of the first samples
# num1:the number of the second samples
# num2:the number of the third samples


fs = open('wine_data.csv', 'r')
reader = csv.reader(fs)

lines = list(reader)

#制作数据集（训练集样本，测试集样本，训练集标签，测试集标签）
lines = np.array((lines[1:]))
train_data = np.zeros((166, 13), dtype=float)
test_data = np.zeros((12, 13), dtype=float)
train_label = np.zeros((166,), dtype=int)
test_label = np.zeros((12,), dtype=int)

train_count = 0
test_count = 0
for i in range(178):
    for j in range(13):
        if i % 16 != 0:
            train_data[train_count][j] = float(lines[i][j])
        else:
            test_data[test_count][j] = float(lines[i][j])
    if i % 16 == 0:
        test_label[test_count] = int(lines[i][13])
        test_count += 1
    else:
        train_label[train_count] = int(lines[i][13])
        train_count += 1


# feature selection
# 根据距离的可分性判据，J2 = Sw^(-1)Sb
# normalization
min_max_scalar = preprocessing.MinMaxScaler()
data = min_max_scalar.fit_transform(train_data)  # 训练集归一化

m1, Sw1 = calculate_Sw(data, 0, 55, 13, 1.0/3)
m2, Sw2 = calculate_Sw(data, 55, 121, 13, 1.0/3)
m3, Sw3 = calculate_Sw(data, 121, 166, 13, 1.0/3)

Sw = Sw1 + Sw2 + Sw3
Sw_inv = np.linalg.inv(Sw)

mi = np.array([m1, m2, m3])

m, Sb = calculate_Sb(data, mi, 13, 1.0/3)

J2 = np.dot(Sw_inv, Sb)

#根据J2求特征值和特征向量
feature_value, feature_vecctor = np.linalg.eig(J2)

#提取特征值最大的几个特征值对应的特征
index_low_to_high = np.argsort(feature_value, kind='quicksort')  # 索引按从小到大排序
# 求降维矩阵Wt
Wt = np.zeros((4, 13))
for i in range(4):
    index = index_low_to_high[12-i]
    print(i, ':', feature_value[index])
    Wt[i] = feature_vecctor[index]

data = np.dot(Wt, data.T).T

test_xx = min_max_scalar.transform(test_data).T  #测试数据归一化处理
test_xx = np.dot(Wt, test_xx).T  #降维

accuracy = 0.0

#采用KNN算法，依据K=25时，根据聚类结果判断类别

for num in range(12):   #12个测试数据
    xx = test_xx[num]
    label = test_label[num]
    dist = []
    for data_x in data:
        data_x = data_x.reshape((1, 4))
        dist.append(np.linalg.norm(data_x-xx))

    index = np.argsort(dist)

    count0 = 0.0
    count1 = 0.0
    count2 = 0.0
    i = 0
    while i<25:
        if train_label[index[i]] == 0:
            count0 += 1
        elif train_label[index[i]] == 1:
            count1 += 1
        elif train_label[index[i]] == 2:
            count2 += 1
        i += 1
    
    # print(count0,count1,count2)
    # print(index)
    count0 /= 6
    count1 /= 7
    count2 /= 5

    if count0>=max(count1,count2):
        print('0',end=",")
        if label == 0:
            accuracy += 1
    elif count1>=max(count0,count2):
        print('1',end=",")
        if label == 1:
            accuracy += 1       
    elif count2>max(count1,count0):
        print('2',end=",")
        if label == 2:
            accuracy += 1

#输出准确率：
print()
print("accuracy:",'{:.2f}'.format(accuracy/12*100),"%")