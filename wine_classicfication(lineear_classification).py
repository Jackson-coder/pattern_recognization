import numpy as np
import csv
from sklearn import preprocessing


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


def training(data, label, w0, w1, w2, c):
    flag = True  # 下一轮需要继续迭代
    count = 0
    np.random.seed(77)
    np.random.shuffle(data)
    np.random.seed(77)
    np.random.shuffle(label)

    for i in range(len(data)):
        value0 = np.dot(w0.T, data[i])
        value1 = np.dot(w1.T, data[i])
        value2 = np.dot(w2.T, data[i])
        # print(value0,value1,value2)
        if value0 <= max(value1, value2) and label[i] == 0 :
            w0 += c * data[i]
            w1 -= c * data[i]
            w2 -= c * data[i]
            count += 1

        elif value1 <= max(value0, value2) and label[i] == 1:
            w0 -= c * data[i]
            w1 += c * data[i]
            w2 -= c * data[i]
            count += 1

        elif value2 <= max(value0, value1) and label[i] == 2:
            w0 -= c * data[i]
            w1 -= c * data[i]
            w2 += c * data[i]
            count += 1

    if count < len(data)*0.21:
        print(len(data), count)
        flag = False

    if flag == True:
        w0, w1, w2 = training(data, label, w0, w1, w2, c)

    return w0, w1, w2


fs = open('wine_data.csv', 'r')
reader = csv.reader(fs)

lines = list(reader)


lines = np.array((lines[1:]))
data = np.zeros((178, 13), dtype=float)
label = np.zeros((178,), dtype=int)

for i in range(178):
    for j in range(13):
        data[i][j] = float(lines[i][j])
    label[i] = int(lines[i][13])

# feature selection

m1, Sw1 = calculate_Sw(data, 0, 59, 13, 1.0/3)
m2, Sw2 = calculate_Sw(data, 59, 130, 13, 1.0/3)
m3, Sw3 = calculate_Sw(data, 130, 178, 13, 1.0/3)

Sw = Sw1 + Sw2 + Sw3
Sw_inv = np.linalg.inv(Sw)

mi = np.array([m1, m2, m3])

m, Sb = calculate_Sb(data, mi, 13, 1.0/3)

J2 = np.dot(Sw_inv, Sb)
feature_value, feature_vecctor = np.linalg.eig(J2)

print(feature_value)

index_low_to_high = np.argsort(feature_value, kind='quicksort')  # 索引按从小到大排序
# 求降维矩阵Wt
Wt = np.zeros((3, 13))
for i in range(3):
    index = index_low_to_high[12-i]
    print(i, ':', feature_value[index])
    Wt[i] = feature_vecctor[index]

x = np.dot(Wt, data.T)

# 由感知器算法实现多类判别

# normalization
# x = preprocessing.MinMaxScaler().fit_transform(x.T).T
new_row = np.ones((1, 178))
# standard
x = np.row_stack((x, new_row)).T
# initial parameters
w0 = np.array([0.0, 0.0, 0.0, 0.0])
w1 = np.array([0.0, 0.0, 0.0, 0.0])
w2 = np.array([0.0, 0.0, 0.0, 0.0])
c = 2
# # 迭代训练
w0, w1, w2 = training(x, label, w0, w1, w2, c)
print(w0, w1, w2)
# value0 = np.dot(w0.T, x[96])
# value1 = np.dot(w1.T, x[96])
# value2 = np.dot(w2.T, x[96])
# print(">",value0,value1,value2)

W = np.array([w0,w1,w2])


# 前向传播
# a = np.array([[13.24,2.59,2.87,21,118,2.8,2.69,0.39,1.82,4.32,1.04,2.93,735]]).T
# a = np.dot(Wt, a)
# a = np.row_stack((a,np.ones(1,)))
# v = np.dot(W,a)
# print(v)



x = np.array([[0.0,0.0,1.0],[1.0,1.0,1.0],[-1.0,1.0,1.0]])
label = np.array([0,1,2])
w0,w1,w2 = training(x,label,np.array([0.0,0.0,0.0]),np.array([0.0,0.0,0.0]),np.array([0.0,0.0,0.0]),1)
print(w0,w1,w2)


# x = np.array([[1.0, 0.0], [2.0, 0.0], [1.0, 1.0],
#               [-1.0, 0.0], [-1.0, 1.0], [0.0, 1.0]])

# m1, Sw1 = calculate_Sw(x, 0, 3, 2, 1.0/2)
# m2, Sw2 = calculate_Sw(x, 3, 6, 2, 1.0/2)
# Sw = Sw1 + Sw2
# print(np.linalg.inv(Sw))

# mi = np.array([m1,m2])
# Sb = calculate_Sb(x,mi,2,1.0/2)
# print(Sb)
