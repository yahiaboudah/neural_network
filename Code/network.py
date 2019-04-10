import numpy as np

class Network(object):

    def __init__(self,size,training_data,testing_data):
        self.num_layers = len(size)
        self.training_data = training_data
        self.testing_data = testing_data
        self.weights = [np.random.random((x,y)) for y,x in zip(size[:-1],size[1:])]
        self.biases = [np.random.random((x,1)) for x in size[1:]]

    def feed_forward(self,a):
        for i in range(self.num_layers-1):
            a = sigmoid(np.dot(self.weights[i],a))
        return a

    def backprop(self,delta):
        #Get the gradients
        return dw,db

    def update_given_mini_batch(self,mini_batch):
        #Take a mini_batch,
        #do forprop,backprop for len(mini_batch),then update params
        self.weights = new_weights
        self.biases = new_biases

    def stochastic_gradient_descent(self,num_epochs):
        for i in range(num_epochs):
            #Create a list mini_batches here
            for mini_batch in mini_batches: #Loop through the data
                update_given_mini_batch(mini_batch)

            #Get the testing accuracy+the training cost:
            print(testing_accuracy())
            print(training_cost())

    def testing_accuracy():
        #Loop through testing_data
