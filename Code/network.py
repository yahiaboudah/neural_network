import numpy as np

class Network(object):

    def __init__(self,size,training_data,testing_data):
        self.num_layers = len(size)
        self.training_data = training_data
        self.testing_data = testing_data
        self.weights = [np.random.random((x,y)) for y,x in zip(size[:-1],size[1:])]
        self.biases = [np.random.random((x,1)) for x in size[1:]]

    def feed_forward(self,a,all_layers = False):
        if(all_layers):
            a = [a]
            for i in range(1,self.num_layers):
                a[i] = sigmoid(np.dot(self.weights[i-1],a[i-1]))
        else:
            for i in range(self.num_layers-1):
                a = sigmoid(np.dot(self.weights[i],a))
        return a


    def backprop(self,a,y):
        #Initialize empty matrices:
        jacob_w = [np.zeros((x,y)) for y,x in \
                                    zip(self.size[:-1],self.size[1:])]
        jacob_b = [np.zeros((x,1)) for x in self.size[1:]]

        #Caclulate the delta of the last layer:
        delta = (a[-1]-y) * sigmoid(a[-1],deriv=True)
        #Update params:
        jacob_w[-1] = np.dot(delta,a[-2].T)
        jacob_b[-1] = delta

        for l in range(2,self.num_layers+1):
            #Get delta of secondtolast layer:
            delta = np.dot(self.weights[-l+1].T, delta) * sigmoid(a[-l],deriv=True)
            #Get derivatives of w and b:
            jacob_b[-l] = delta
            jacob_w[-l] = np.dot(delta,a[-l-1].T)

        #Get the gradients
        return jacob_w,jacob_b


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
        pass
