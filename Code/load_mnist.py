from mnist import MNIST
import numpy as np
import random

def get_data():
    mndata = MNIST('Data')
    training_data = mndata.load_training()
    testing_data = mndata.load_testing()
    return training_data,testing_data

#import matplotlib.pyplot as plt
"""
index = random.randrange(0,len(testing_data))
arr = (np.mat(np.array(training_data[index]))).reshape(28,28)
plt.imshow(arr,cmap='binary')
plt.show()
"""
