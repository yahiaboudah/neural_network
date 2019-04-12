from mnist import MNIST
import numpy as np
import random
import matplotlib.pyplot as plt
import timeit

def load_data():
    mndata = MNIST('Data')
    loaded_training_data = mndata.load_training()
    loaded_testing_data = mndata.load_testing()
    return loaded_training_data,loaded_testing_data

def get_data():
    loaded_training_data,loaded_testing_data = load_data()
    training_inputs = [np.divide(np.reshape(x, (784, 1)),255) for x in loaded_training_data[0]]
    training_labels = [vectorize(b) for b in loaded_training_data[1]]
    training_data = list(zip(training_inputs,training_labels))
    testing_inputs = [np.divide(np.reshape(x,(784,1)),255) for x in loaded_testing_data[0]]
    testing_data = list(zip(testing_inputs,loaded_testing_data[1]))

    return training_data,testing_data

def vectorize(b):
    m = np.zeros((10,1))
    m[b] = 1
    return m

"""
print(timeit.timeit('get_data()',number=1,setup= 'from __main__ import get_data'))
training_data,testing_data = get_data()
index = random.randrange(0,len(training_data))
arr = (np.mat(training_data[index][0])).reshape(28,28)
plt.imshow(arr,cmap='binary')
plt.show()
"""
