import numpy as np

class Network(object):

    def __init__(self,size,training_data,testing_data):
        self.num_layers = len(size)
        self.training_data = training_data
        self.testing_data = testing_data
        self.weights = [np.random.random((x,y)) for y,x in zip(size[:-1],size[1:])]
        self.biases = [np.random.random((x,1)) for x in size[1:]]

    def feed_forward(self,a,all_layers = True):
        if(all_layers):
            a = [a]
            for i in range(1,self.num_layers):
                a[i] = sigmoid(np.dot(self.weights[i-1]+self.biases[i-1],a[i-1]))
        else:
            for i in range(self.num_layers-1):
                a = sigmoid(np.dot(self.weights[i]+self.biases[i],a))
        return a

    def backprop(self,a,y,dw,db):
        #Caclulate the delta of the last layer:
        delta = (a[-1]-y) * sigmoid(a[-1],deriv=True)
        #Update params:
        dw[-1] += (np.dot(delta,a[-2].T))/mini_batch_size
        db[-1] += (delta)/mini_batch_size

        for l in range(2,self.num_layers+1):
            #Get delta of secondtolast layer:
            delta = np.dot(self.weights[-l+1].T, delta) * sigmoid(a[-l],deriv=True)
            db[-l] += delta/mini_batch_size
            dw[-l] += (np.dot(delta,a[-l-1].T))/mini_batch_size
        #Return the matrices
        return dw,db


    def update_given_mini_batch(self,mini_batch):
        #Create two empty matrices to store derivatives:
        dw = [np.zeros((x,y)) for y,x in zip(self.size[:-1],self.size[1:])]
        db = [np.zeros((x,1)) for x in self.size[1:]]
        #Go through the training data to get derivatives:
        for x,y in mini_batch:
            a = self.feed_forward(x)
            dw,db = self.backprop(a,y,dw,db)
        #Now update the weights and biases given dw,db:
        for i in range(num_layers-1):
            self.weights[i] -= learning_rate*dw[i]
            self.biases[i] -= learning_rate*db[i]

    def stochastic_gradient_descent(self,num_epochs):
        for i in range(num_epochs):
            #Create a list of mini_batches here
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in range(0,len(training_data),mini_batch_size)]
            for mini_batch in mini_batches: #Loop through the data
                update_given_mini_batch(mini_batch)

            print('Testing Accuracy:', self.get_training_accuracy(),'/10000')

    def get_training_accuracy():
        for x in training_data:
            a = feed_forward(x)





q
