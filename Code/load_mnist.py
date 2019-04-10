from mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np
import random

mndata = MNIST('Data')
training_data, training_labels = mndata.load_training()
testing_data, testing_labels = mndata.load_testing()

index = random.randrange(0,len(testing_data))
arr = (np.mat(np.array(training_data[index]))).reshape(28,28)
plt.imshow(arr,cmap='binary')
plt.show()
