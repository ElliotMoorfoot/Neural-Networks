import random
import math
import numpy as np
import matplotlib.pyplot as plt



class Network(object):#class of networks
    
    def __init__(self,size):
        self.num_layers = len(size)
        self.size = size
        self.biases = [np.zeros((y,1)) for y in size[1:]]
        self.weights = [((1/x)**0.5)*np.random.randn(y, x) for x,y in zip(size[:-1], size[1:])]
        self.batch_num = 1
        self.epoch = 1
        self.data = []
       


#net = network([2,3,4]) makes a network with layers with sizes 2 , 3 and 4 all with random wieghts and bases between 0 and 1 actually normal dist
    
    def shuffle(self):
        new_data_x = []
        new_data_y = []
        random.shuffle(self.data)
        for x in self.data:
            new_data_x.append(x[0])
            new_data_y.append(x[1])
        return np.array(new_data_x),np.transpose(np.array(new_data_y))
    

    
    
    def output(self,a):#output of network

        for w, b in zip(self.weights,self.biases):
            z = (w@a) + b
            a = relu(z)
        return a
    
    def gradient(self,a,y):#calculates gradient of cost func with sample a using backpropagation
        # y here is just a number need it in vector form
        inputs = []
        m = len(a)
        a = np.reshape(a,(m,784))
        a = np.transpose(a)
        activations = [a] 
        '''need to make activations the same shape as the weights'''
        for w, b in zip(self.weights,self.biases):
            z = (w@a) + b
            a = relu(z)
            inputs.append(z)
            activations.append(a)
        '''need y = desired output of network'''
        '''(activations[-1] - y ) is the derv of the cost funct (quadratic here)'''
        error = a-y 

        errors = [error]
        for i in range(1,self.num_layers-1):
            error = (np.transpose(self.weights[-i]) @ errors[0]) * relu_prime(inputs[-(i+1)])
            errors.insert(0, error)
        activations = activations[:-1]
        grad_w = [x@np.transpose(y) for x,y in zip(errors,activations)]
        return errors, grad_w
    
    def learn(self,sample_x,sample_y,LR,n,WD):#uses a training sample to calculate the grad of c and improve the network using grad descent
        # LR = learning rate
        # WD = weight decay constant
       
        m = len(sample_x)
        sum_w = [np.zeros(i.shape) for i in self.weights]
        sum_b = [np.zeros(i.shape) for i in self.biases]
        


        grad_b,grad_w = self.gradient(sample_x,sample_y)
        sum_b = [np.sum(x,axis=1,keepdims = True) for x in grad_b]
        sum_w = grad_w 
        

        self.weights = [w - (LR/m)*x for w,x in zip(self.weights,sum_w)]
        self.biases = [b - (LR/m)*x for b,x in zip(self.biases,sum_b)]
        

        
       
    def train(self,data_x,data_y,LR,size,epochs,drop,WD=0.1,shuffle=False):#need an epoch/time agruement? need size|60000
        '''LR is learning rate
        plot determines whether we plot the cost fct against epoch
        size is the number of data used for one back propagation/gradient descent
        epochs is the number of time we train the net with the training data
        WD is weight decay (L2 reg)
        drop is probability that each node/weight? is ignored for dropout
        we should shuffle the data each time?'''
        costs = []
        epoch = []
        n = len(data_y)
        num_y = data_y 
        m = int(n/size) #m is the total number of times we go through learn.
        vect = np.zeros((10,n)) 
        for i in range(n):
            vect[data_y[i][0]][i] = 1.0
        data_y = vect # the y data in vectorised form
        self.data = [(i,j) for i,j in zip(data_x,np.transpose(data_y))]
        for j in range(epochs):
            if shuffle:
                data_x,data_y = self.shuffle()
            for i in range(0,n,size):
                self.learn(data_x[i:i+size],data_y[:,i:i+size],LR,n,WD)#seems decent enough
                self.batch_num += 1
            self.epoch += 1
        if plot:
            plt.plot(epoch,costs,'-m',label='Training Set')
            plt.xlabel("Epoch")
            plt.ylabel("Cost")
            plt.show()
        
        
    def test(self,test_x,test_y):#returns the number of digits correctly identified by the network and the total number of digits tested
        tot = len(test_y)
        count = 0
        for x,y in zip(self.output(test_x),test_y):
            if y == np.argmax(x):
                count += 1
        return(count,tot)
    
def relu(z):# z = w*x + b
    x = np.zeros(np.shape(z))
    return np.maximum(z,x)

def relu_prime(z):
    return (z>0).astype(z.dtype)
    
def sigmoid(z):# z = w*x + b
    return 1/(1 + np.exp(-z))

def sigmoid_prime(z):
    x = sigmoid(z)
    return x*(1-x)

def softmax(z):
    """Returns the value of the Softmax function"""
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)