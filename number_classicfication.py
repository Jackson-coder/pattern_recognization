import numpy as np
import os
from sklearn import preprocessing
from sklearn import svm
from sklearn import decomposition,discriminant_analysis
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, make_scorer,recall_score
from cv2 import cv2 as cv
from sklearn.model_selection import cross_val_score,cross_validate
import matplotlib.pyplot as plt

# 数据集制作：
# @path_name:"the path of the picture file"
# @return training_samples,training_labels,testing_samples,testing_labels


def data_process(path_name):
    samples = []
    labels = []

    for root_paths, sub_dirs_names, files in os.walk(path_name):
        number = str(os.path.split(root_paths)[1][-1])
        number = (ord(number)-ord('0')+9) % 10
        for File in files:
            file_png = cv.imread(root_paths+'/'+File)
            file_png = cv.cvtColor(file_png, cv.COLOR_BGR2GRAY)
            # 将图片缩小化，实现数据量的大幅减少，减少冗余数据对分类效果、分类时长的不良影响
            file_png = cv.resize(file_png, (32, 24))
            # 将图片数据进行归一化处理，flatten成一维数据，便于后续的特征提取操作
            file_png = file_png/255
            data = np.reshape(file_png, (1, -1))
            # 将图片数据和对应的标签相匹配
            samples.append(data)
            labels.append(number)
            # print(data)

    samples = np.array(samples)
    samples = np.resize(samples, (550, 768))

    # LDA降维，提取相关性较强的特征

    # pca = decomposition.PCA(n_components=40)
    lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=9)
    new_samples = lda.fit_transform(samples,labels)
    # print(pca.explained_variance_ratio_)

    # 设置随机种子，将前500个样本作为训练集，将后50个样本作为测试集

    np.random.seed(80)
    np.random.shuffle(new_samples)
    np.random.seed(80)
    np.random.shuffle(labels)

    training_samples = np.array(new_samples[:500])
    testing_samples = np.array(new_samples[500:])
    training_labels = np.array(labels[:500])
    testing_labels = np.array(labels[500:])

    return training_samples, training_labels, testing_samples, testing_labels


# 导入训练样本，训练标签，测试样本，测试标签
training_samples, training_labels, testing_samples, testing_labels = data_process(
    'Img')


training_samples = np.reshape(training_samples, (500, -1))
testing_samples = np.reshape(testing_samples, (50, -1))
print(training_samples.shape)

# # 使用网格搜索法，选择非线性SVM“类”中的最佳C值
# kernel = ['linear', 'rbf', 'sigmoid']
# C = [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 9, 10]  # 5
# parameters = {'kernel': kernel, 'C': C}
# grid_svc = model_selection.GridSearchCV(estimator=svm.SVC(decision_function_shape='ovo'
# ), param_grid=parameters, scoring='accuracy', cv=5, verbose=1)
# print("waiting...")
# # 模型在训练数据集上的拟合
# grid_svc.fit(training_samples, training_labels)
# # 返回交叉验证后的最佳参数值
# print(grid_svc.best_params_, grid_svc.best_score_)

# # svm_svc = svm.SVC(C=1,kernel='rbf')
# svm_svc = svm.SVC(**(grid_svc.best_params_),decision_function_shape='ovo')

svm_svc = svm.SVC(C=1,decision_function_shape='ovo')

svm_svc.fit(training_samples, training_labels)

# 模型在测试集上的预测
pred_svc = svm_svc.predict(testing_samples)

# 模型的预测准确率
scoring = ['precision_macro', 'recall_macro']
#print(cross_val_score(svm_svc, training_samples, training_labels, cv=5))
#通过设置return_estimator=True来保留在所有训练集上拟合好的估计器;
#return_train_score 默认设置为 True 。 它增加了所有 scorers(得分器) 的训练得分 keys 。如果不需要训练 scores ，则应将其明确设置为 False 
scores = cross_validate(svm_svc, training_samples, training_labels, scoring=scoring, cv=50, return_train_score=True)

t = np.vstack((scores['train_recall_macro'],scores['train_precision_macro']))
t = t.T[np.argsort(t.T[:,0])].T

# svm_clu = svm.SVR(kernel='rbf')
# svm_clu.fit(scores['train_recall_macro'].reshape((-1,1)), scores['train_precision_macro'])
# pred_svc = svm_clu.predict(np.random.uniform(0,1,(10,)).reshape((-1,1)))

plt.plot(t[0],t[1])
plt.yticks(np.linspace(0.9,1.0,10))
plt.ylabel('precision')
plt.xlabel('recall')
plt.title('PR')
plt.show()

print("training_accuracy", svm_svc.score(training_samples, training_labels)*100,"%")
print("testing_accuracy", svm_svc.score(testing_samples, testing_labels)*100,"%")


# # 测试样本真实标签和预测标签
# # print(testing_labels)
# # print(result)
