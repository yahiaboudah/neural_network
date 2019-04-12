import numpy as np
import random as random
import load_mnist as mn

class Network(object):

    def __init__(self,size,training_data,testing_data,mini_batch_size,num_epochs,learning_rate):
        self.num_layers = len(size)
        self.learning_rate = learning_rate
        self.size = size
        self.mini_batch_size = mini_batch_size
        self.weights = [np.random.random((x,y)) for y,x in zip(self.size[:-1],self.size[1:])]
        self.biases = [np.random.random((x,1)) for x in self.size[1:]]
        self.num_epochs = num_epochs

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
        delta = (a[-1]-y)
        #Update params:
        dw[-1] += ((np.dot(delta,a[-2].T))/self.mini_batch_size)
        db[-1] += ((delta)/self.mini_batch_size)

        for l in range(2,self.num_layers):#num_layers+1 previously
            #Get delta of secondtolast layer:
            delta = np.dot(self.weights[-l+1].T, delta) * self.sigmoid_deriv(a[-l])
            db[-l] += (delta/self.mini_batch_size)
            dw[-l] += ((np.dot(delta,a[-l-1].T))/self.mini_batch_size)
        #Return the matrices
        return dw,db


    def update_given_mini_batch(self,mini_batch):
        #Create two empty matrices to store derivatives:
        dw = [np.zeros(w.shape) for w in self.weights]
        db = [np.zeros(b.shape) for b in self.biases]
        #Go through the training data to get derivatives:
        for x,y in mini_batch:
            a = self.feed_forward(x)
            dw,db = self.backprop(a,y,dw,db)
        #Now update the weights and biases given dw,db:
        for i in range(self.num_layers-1):
            self.weights[i] -= self.learning_rate*dw[i]
            self.biases[i] -= self.learning_rate*db[i]

    def stochastic_gradient_descent(self):
        n = len(training_data)
        random.shuffle(training_data)
        for i in range(self.num_epochs):
            #Create a list of mini_batches here
            mini_batches = [training_data[k:k+self.mini_batch_size] for k in range(0,n,self.mini_batch_size)]
            for mini_batch in mini_batches: #Loop through the data
                self.update_given_mini_batch(mini_batch)
            print('Epoch ',(i+1),': training complete.')
            print('Testing Accuracy:', self.get_training_accuracy(),'/10000')

    def get_training_accuracy(self):
        i = 0
        for x,y in training_data:
            i+=1
            if(i % 1000):
                a = self.feed_forward(x,all_layers=False)
                print(a)
        res = [(np.argmax(self.feed_forward(x,all_layers=False)),np.argmax(y)) for x,y in training_data]
        return sum(int(x == y) for (x,y) in res)

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def sigmoid_deriv(self,x):
        return x * (1-x)


training_data,testing_data = mn.get_data()

net = Network([784,30,10],training_data,testing_data,20,5,1.5)
net.stochastic_gradient_descent()
