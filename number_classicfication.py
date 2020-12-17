import numpy as np
import os
from sklearn import preprocessing
from sklearn import svm
from sklearn import model_selection
from cv2 import cv2 as cv

def data_process(path_name):
    samples = []
    labels = []

    for root_paths,sub_dirs_names,files in os.walk(path_name):
        number = str(os.path.split(root_paths)[1][-1])
        number = (ord(number)-ord('0')+9)%10
        for File in files:
            file_png = cv.imread(root_paths+'/'+File)
            file_png = cv.cvtColor(file_png,cv.COLOR_BGR2GRAY)
            file_png = cv.resize(file_png,(30,40))
            # cv.imshow("png",file_png)
            # cv.waitKey(0)
            data = np.reshape(file_png,(1,-1))
            samples.append(data)
            labels.append(number)
            print(root_paths)
            
    np.random.seed(90)
    np.random.shuffle(samples)
    np.random.seed(90)
    np.random.shuffle(labels)

    training_samples = np.array(samples[:500])
    testing_samples = np.array(samples[500:])
    training_labels = np.array(labels[:500])
    testing_labels = np.array(labels[500:])

    return training_samples,training_labels,testing_samples,testing_labels

training_samples,training_labels,testing_samples,testing_labels = data_process('Img')

training_samples = np.reshape(training_samples,(500,-1))
testing_samples = np.reshape(testing_samples,(50,-1))
print(training_samples.shape)


svc = svm.SVC()
param = {'C':[1,2,3,4,5]}
grid = model_selection.GridSearchCV(svc,param,scoring='accuracy')
grid.fit(training_samples,training_labels)
best_params,best_score = grid.best_params_,grid.best_score_

# best_params = {'C': 6, 'kernel': 'rbf'}
print(best_params)
svc = svm.SVC(**best_params)

svc.fit(training_samples,training_labels)
print(svc.score(training_samples,training_labels))
print(svc.score(testing_samples,testing_labels))
result = svc.predict(testing_samples)

print(testing_labels)
print(result)