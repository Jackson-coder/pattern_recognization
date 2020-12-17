import numpy as np
import os
from sklearn import preprocessing
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
            print(file_png.shape)
            file_png = cv.resize(file_png,(300,400))
            # # cv.imshow("png",file_png)
            cv.waitKey(0)
            samples.append(file_png)
            labels.append(number)
            print(root_paths)
            
    np.random.seed(77)
    np.random.shuffle(samples)
    np.random.seed(77)
    np.random.shuffle(labels)

    training_samples = samples[:500]
    testing_samples = samples[500:]
    training_labels = labels[:500]
    testing_labels = labels[500:]

    return (training_samples,training_labels),(testing_samples,testing_labels)

training, testing = data_process('Img')
print(training,testing)



