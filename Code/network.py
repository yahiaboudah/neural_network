import numpy as np
import random as random
import load_mnist as mn

class Network(object):

    def __init__(self,size):
        self.num_layers = len(size)
        self.size = size
        self.weights = [np.random.randn(x,y) for y,x in zip(size[:-1],size[1:])]
        self.biases = [np.random.randn(x,1) for x in size[1:]]

    def feed_forward(self,a,all_layers = True):

        if(all_layers):
            a = [a]
            for i in range(1,self.num_layers):
                a.append(self.sigmoid(np.dot(self.weights[i-1],a[i-1])+self.biases[i-1]))

        else:
            for b,w in zip(self.biases,self.weights):
                a = self.sigmoid(np.dot(w,a)+b)

        return a

    def backprop(self,a,y,dw,db):
        #Caclulate the delta of the last layer:
        delta = (a[-1]-y) * self.sigmoid_deriv(a[-1])

        #Update params:
        dw[-1] += ((np.dot(delta,a[-2].T)))
        db[-1] += ((delta))

        for l in range(2,self.num_layers):#num_layers+1 previously
            #Get delta of secondtolast layer:
            delta = np.dot(self.weights[-l+1].T, delta) * self.sigmoid_deriv(a[-l])
            db[-l] += (delta)
            dw[-l] += ((np.dot(delta,a[-l-1].T)))

        #Return the matrices
        return dw,db


    def update_given_mini_batch(self,mini_batch,learning_rate):
        mini_batch_size = float(len(mini_batch))
        #Create two empty matrices to store derivatives:
        dw = [np.zeros(w.shape) for w in self.weights]
        db = [np.zeros(b.shape) for b in self.biases]

        #Go through the training data to get derivatives:
        for x,y in mini_batch:
            a = self.feed_forward(x)
            dw,db = self.backprop(a,y,dw,db)

        #Now update the weights and biases given dw,db:
        for i in range(self.num_layers-1):
            self.weights[i] -= (learning_rate/mini_batch_size)*dw[i]
            self.biases[i] -= (learning_rate/mini_batch_size)*db[i]

    def SGD(self,training_data,num_epochs,mini_batch_size,learning_rate,test_data=None):

        n = len(training_data)
        n_batches = int(n/mini_batch_size)
        random.shuffle(training_data)

        for j in range(num_epochs):
            for i in range(n_batches): #Loop through the data
                mini_batch = training_data[mini_batch_size*i:mini_batch_size*(i+1)]
                self.update_given_mini_batch(mini_batch,learning_rate)

            print('Epoch ',(j+1),': training complete.')
            print('Testing Accuracy:', self.get_training_accuracy(test_data),'/10000')

    def get_training_accuracy(self,testing_data):
        res = [not bool(np.argmax(self.feed_forward(x,all_layers=False))-y)
                                        for x,y in testing_data]
        return sum(x for x in res)

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def sigmoid_deriv(self,x):
        return x * (1-x)


training_data,testing_data = mn.get_data()


net = Network([784,30,10])
#net.SGD(training_data,1,10,3.0,test_data=testing_data)


import timeit
timeTaken = timeit.repeat('net.SGD(training_data,1,10,3.0,test_data=testing_data)',repeat=2,number=1,setup='from __main__ import Network,net,training_data,testing_data')
print(timeTaken)
