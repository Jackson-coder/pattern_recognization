import numpy as np
import math
import matplotlib.pyplot as plt
# 保存训练样本
samples = np.array([[[1.58, 2.32, -5.8], [0.67, 1.58, -4.78], [1.04, 1.01, -3.63], [-1.49, 2.18, -3.39], [-0.41, 1.21, -4.73],
                     [1.39, 3.16, 2.87], [1.20, 1.40, -1.89], [-0.92, 1.44, -3.22], [0.45, 1.33, -4.38], [-0.76, 0.84, -1.96]],
                    [[0.21, 0.03, -2.21], [0.37, 0.28, -1.8], [0.18, 1.22, 0.16], [-0.24, 0.93, -1.01], [-1.18, 0.39, -0.39],
                     [0.74, 0.96, -1.16], [-0.38, 1.94, -0.48], [0.02, 0.72, -0.17], [0.44, 1.31, -0.14], [0.46, 1.49, 0.68]],
                    [[-1.54, 1.17, 0.64], [5.41, 3.45, -1.33], [1.55, 0.99, 2.69], [1.86, 3.19, 1.51], [1.68, 1.79, -0.87],
                     [3.51, -0.22, -1.39], [1.40, -0.44, -0.92], [0.44, 0.83, 1.97], [0.25, 0.68, -0.99], [0.66, -0.45, 0.08]]])


def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y


def d_sigmoid(x):
    fx = sigmoid(x)
    y = fx*(1-fx)
    return y


def tanh(x):
    y = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    return y


def d_tanh(x):
    sum = 1-tanh(x)**2
    return sum


def mse_loss(x, y):
    sum = np.dot(x, y.T)/2
    return sum


# single netral


class netral():
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def forward_1(self, inputs):
        y = np.dot(self.weight, inputs.T)+self.bias
        y = sigmoid(y)
        return y

    def forward_2(self, inputs):
        y = np.dot(self.weight, inputs.T)+self.bias
        y = tanh(y)
        return y

# netral network

# input layer 3
# hidden layer 3
# output layer 3


class network():
    def __init__(self):
        # initial_parameters
        self.weight_hidden = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]])
        self.weight_output = np.array([[1, 1, 0], [0, 1, 0], [1, 0, 1]])
        self.bias_hidden = np.array([0, 0, 0])
        self.bias_output = np.array([0, 0, 0])

        self.hidden_1 = netral(self.weight_hidden[0], self.bias_hidden[0])
        self.hidden_2 = netral(self.weight_hidden[1], self.bias_hidden[1])
        self.hidden_3 = netral(self.weight_hidden[2], self.bias_hidden[2])

        self.output_1 = netral(self.weight_output[0], self.bias_output[0])
        self.output_2 = netral(self.weight_output[1], self.bias_output[1])
        self.output_3 = netral(self.weight_output[2], self.bias_output[2])


def predict(inputs, n):
    inputs = inputs / \
        math.sqrt(inputs[0]*inputs[0]+inputs[1]*inputs[1]+inputs[2]*inputs[2])
    hidden_y1 = n.hidden_1.forward_2(inputs)
    hidden_y2 = n.hidden_2.forward_2(inputs)
    hidden_y3 = n.hidden_3.forward_2(inputs)
    hidden_y = np.array([hidden_y1, hidden_y2, hidden_y3])

    output_y1 = n.output_1.forward_1(hidden_y)
    output_y2 = n.output_2.forward_1(hidden_y)
    output_y3 = n.output_3.forward_1(hidden_y)
    output_real = np.array([output_y1, output_y2, output_y3])

    max_num = np.argmax(output_real)
    return max_num

score = []
step = []

def training(epoches):
    n = network()
    a = 0.1  # learning_rate

    for epoch in range(epoches):
        print('epoch:', epoch,end=',')
        label = -1
        # print(n.weight_output)

        for sample in samples:
            label += 1
            index = label
            output = [0, 0, 0]
            output[index] = 1

            for i in range(6):
                inputs = sample[i]
            # for inputs in sample:

                # forward
                # normlization
                inputs = inputs / \
                    math.sqrt(inputs[0]*inputs[0]+inputs[1]
                              * inputs[1]+inputs[2]*inputs[2])

                n.hidden_1.weight = n.weight_hidden[0]
                n.hidden_2.weight = n.weight_hidden[1]
                n.hidden_3.weight = n.weight_hidden[2]
                n.output_1.weight = n.weight_output[0]
                n.output_2.weight = n.weight_output[1]
                n.output_3.weight = n.weight_output[2]

                hidden_y1 = n.hidden_1.forward_2(inputs)
                hidden_y2 = n.hidden_2.forward_2(inputs)
                hidden_y3 = n.hidden_3.forward_2(inputs)
                hidden_y = np.array([hidden_y1, hidden_y2, hidden_y3])

                output_y1 = n.output_1.forward_1(hidden_y)
                output_y2 = n.output_2.forward_1(hidden_y)
                output_y3 = n.output_3.forward_1(hidden_y)
                output_real = np.array([output_y1, output_y2, output_y3])

                # loss = mse_loss(output, output_real)

                # backward

                # if i==0 or i==4:
                #     total = output-output_real
                # elif i!=3 and i!=7 :
                #     total = total+output-output_real
                #     continue
                # else :
                #     total = total+output-output_real
                #     total /= 4

                d_loss_d_output_real = output-output_real
                # d_loss_d_output_real = total

                # output_layer
                d_sigmoid = np.multiply(
                    output_real, 1-output_real).reshape(1, 3)

                d_output_real_d_output_weight = d_sigmoid.T * hidden_y

                d_output_real_d_output_bias = d_sigmoid

                # print(d_sigmoid)
                # print(hidden_y)
                # print(d_output_real_d_output_weight)
                # print(d_loss_d_output_real)
                # print(d_output_real_d_output_bias)

                d_loss_d_output_weight = np.zeros(shape=(3, 3))
                d_loss_d_output_bias = np.zeros(shape=(3,))
                d_output_real_d_hidden_y = np.zeros(shape=(3, 3))
                for i in range(3):
                    d_loss_d_output_bias[i] = d_loss_d_output_real[i] * \
                        d_output_real_d_output_bias[0][i]
                    for j in range(3):
                        d_loss_d_output_weight[i][j] = d_loss_d_output_real[i] * \
                            d_output_real_d_output_weight[i][j]
                        d_output_real_d_hidden_y[i][j] = d_sigmoid[0][i] * \
                            n.weight_output[i][j]
                n.weight_output = n.weight_output + a*d_loss_d_output_weight
                n.bias_output = n.bias_output + a*d_loss_d_output_bias

                # hidden_layer
                d_tanh = (1 - np.multiply(hidden_y, hidden_y)).reshape(1, 3)

                d_hidden_y_d_hidden_weight = d_tanh.T * inputs
                d_hidden_y_d_hidden_bias = d_tanh
                d_loss_d_hidden_weight = np.zeros(shape=(3, 3))
                d_loss_d_hidden_bias = np.zeros(shape=(3,))
                d_loss_d_hidden_y = np.zeros(shape=(3,))
                for i in range(3):
                    d_loss_d_hidden_bias[i] = d_loss_d_hidden_y[i] * \
                        d_hidden_y_d_hidden_bias[0][i]
                    for j in range(3):
                        d_loss_d_hidden_y[i] += d_loss_d_output_real[i] * \
                            d_output_real_d_hidden_y[i][j]

                # print(d_hidden_y_d_hidden_weight)
                for i in range(3):
                    for j in range(3):
                        d_loss_d_hidden_weight[i][j] = d_loss_d_hidden_y[i] * \
                            d_hidden_y_d_hidden_weight[i][j]

                n.weight_hidden = n.weight_hidden + a*d_loss_d_hidden_weight
                n.bias_hidden = n.bias_hidden + a*d_loss_d_hidden_bias
        count=0
        label=-1
        for sample in samples:
            label += 1
            for i in range(4):
                inputs = sample[i+6]
                num = predict(np.array(inputs), n)

                if num == label:
                    count+=1
        print(count/12,num,label)
        step.append(epoch)
        score.append(count/12)

    return n


# output_gradient
# # 保存训练样本的标签
# sample1_label = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
#                           [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]])
# sample2_label = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
#                           [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]])
# sample3_label = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
#                           [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]])
epoches = 8000
n = training(epoches)
# predict(np.array([-1.54, 1.17, 0.64]),n)
label = -1
for sample in samples:
    label += 1
    index = label
    output = [0, 0, 0]
    output[index] = 1
    for inputs in sample:
        print(output, end=',')
        predict(np.array(inputs), n)


# ([[[1.58, 2.32, -5.8], [0.67, 1.58, -4.78], [1.04, 1.01, -3.63], [-1.49, 2.18, -3.39], [-0.41, 1.21, -4.73],
#     [1.39, 3.16, 2.87], [1.20, 1.40, -1.89], [-0.92, 1.44, -3.22], [0.45, 1.33, -4.38], [-0.76, 0.84, -1.96]],
# [[0.21, 0.03, -2.21], [0.37, 0.28, -1.8], [0.18, 1.22, 0.16], [-0.24, 0.93, -1.01], [-1.18, 0.39, -0.39],
#     [0.74, 0.96, -1.16], [-0.38, 1.94, -0.48], [0.02, 0.72, -0.17], [0.44, 1.31, -0.14], [0.46, 1.49, 0.68]],
# [[-1.54, 1.17, 0.64], [5.41, 3.45, -1.33], [1.55, 0.99, 2.69], [1.86, 3.19, 1.51], [1.68, 1.79, -0.87],
#     [3.51, -0.22, -1.39], [1.40, -0.44, -0.92], [0.44, 0.83, 1.97], [0.25, 0.68, -0.99], [0.66, -0.45, 0.08]]])

f = np.polyfit(step, score, 4)
p = np.poly1d(f)
score = p(step)

plt.figure()
plt.plot(step,score,linestyle='dashed',linewidth=0.5,color='red',marker='.',label='square line')
plt.legend()
plt.ylim(0,1)
plt.xlim(0,8000)
ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_position(('data',0))
ax.spines['bottom'].set_position(('data',0))
plt.show()