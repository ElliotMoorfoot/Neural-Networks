import random
import numpy as np


class Network(object):#class of networks
    
    def __init__(self,size):
        self.num_layers = len(size)
        self.size = size
        self.biases = [np.zeros((y,1)) for y in size[1:]]
        self.weights = [((1/x)**0.5)*np.random.randn(y, x) for x,y in zip(size[:-1], size[1:])]#use zip in output or dont use it here
        self.mom_w = [np.zeros(i.shape) for i in self.weights]
        self.mom_b = [np.zeros(i.shape) for i in self.biases]
        self.RMS_w = [np.zeros(i.shape) for i in self.weights]
        self.RMS_b = [np.zeros(i.shape) for i in self.biases]
        self.mom_wc = [np.zeros(i.shape) for i in self.weights]
        self.mom_bc = [np.zeros(i.shape) for i in self.biases]
        self.RMS_wc = [np.zeros(i.shape) for i in self.weights]
        self.RMS_bc = [np.zeros(i.shape) for i in self.biases]
        self.batch_num = 1
        self.epoch = 1



     
   
    
    def output(self,a):
        for w, b in zip(self.weights,self.biases):
            z = (w@a) + b
            a = relu(z)
        return a
    
    def gradient(self,a,y):#calculates gradient of cost func with sample a using backpropagation
        inputs = []
        m = len(y)
        a = np.reshape(a,(m,784))
        a = np.transpose(a)
        activations = [a] 
        vect = np.zeros((10,m))
        for i in range(m):
            vect[y[i][0]][i] = 1.0
        y = vect
        for w, b in zip(self.dropped,self.biases):
            z = (w@a) + b
            inputs.append(z)
            activations.append(a)
        error = a - y
        errors = [error] 
        for i in range(1,self.num_layers-1):
            error = (np.transpose(self.dropped[-i]) @ errors[0] ) * relu_prime(inputs[-(i+1)])
            errors.insert(0, error)
        activations = activations[:-1]
        grad_w = [x@np.transpose(y) for x,y in zip(errors,activations)]
        '''these are exactly the same ^^ '''
        return errors, grad_w
    
    def learn(self,sample_x,sample_y,LR,n,WD):#uses a training sample to calculate the grad of c and improve the network using grad descent
        # LR = learning rate
        # WD = weight decay constant
        '''
        here i calculate all the gradients then average over the mini batch. i can average the error then back propagate that instead saving time
        also need softmax still
        '''
        m = len(sample_y)
        sum_w = [np.zeros(i.shape) for i in self.weights]
        sum_b = [np.zeros(i.shape) for i in self.biases]
        


        grad_b,grad_w = self.gradient(sample_x,sample_y)
        sum_b = [np.sum(x,axis=1,keepdims = True) for x in grad_b]
        sum_w = grad_w 
        
        #LR/self.epoch**0.5
        #LR*0.95**self.epoch
        self.weights = [(1-((LR*0.95**self.epoch)*WD)/m)*w - ((LR*0.95**self.epoch)/m)*x for w,x in zip(self.weights,sum_w)]#this is fine i think
        self.biases = [b - ((LR*0.95**self.epoch)/m)*x for b,x in zip(self.biases,sum_b)]
        
        #self.mom_w = [0.9*y + 0.1*x for x,y in zip(sum_w,self.mom_w)]
        #self.mom_b = [0.9*y + 0.1*x for x,y in zip(sum_b,self.mom_b)]
        #self.RMS_w = [0.999*y + 0.001*np.square(x) for x,y in zip(sum_w,self.RMS_w)]#check this is element wise and not just **2
        #self.RMS_b = [0.999*y + 0.001*np.square(x) for x,y in zip(sum_b,self.RMS_b)]
    #    
     #   self.mom_wc = [i*(1/(1-0.9**self.batch_num)) for i in self.mom_w]
      #  self.mom_bc = [i*(1/(1-0.9**self.batch_num)) for i in self.mom_b]
       # self.RMS_wc = [i*(1/(1-0.999**self.batch_num)) for i in self.RMS_w]
        #self.RMS_bc = [i*(1/(1-0.999**self.batch_num)) for i in self.RMS_b]
        
        #self.weights = [(1-((LR/self.epoch**0.5)*WD)/n)*w - ((LR/self.epoch**0.5)/m)*(x/(y**0.5 + 1e-8)) for w,x,y in zip(self.weights,self.mom_wc,self.RMS_wc)]#this is fine i think
        #self.biases = [b - ((LR/self.epoch**0.5)/m)*(x/(y**0.5 + 1e-8)) for b,x,y in zip(self.biases,self.mom_bc,self.RMS_bc)]
        
    def train(self,data_x,data_y,LR,size,times,drop,WD=0.1):#need an epoch/time agruement? need size|60000
        '''LR is learning rate
        size is the number of data used for one back propagation/gradient descent
        times is the number of time we train the net with the training data
        WD is weight decay (L2 reg)
        drop is probability that each weight is ignored for dropout
        '''
        for j in range(times):
            n = len(data_y)
            m = int(n/size) 
            for i in range(0,n,size):
                if drop == 0:
                    self.dropped = self.weights
                else:
                    dropout = [np.round((0.5/drop)*np.random.uniform(0,1,k.shape)) for k in self.weights[1:-1]]
                    self.dropped = [self.weights[0]]
                    for x,y in zip(dropout,self.weights[1:-1]):
                        self.dropped.append(x*y)
                    self.dropped.append(self.weights[-1])
                self.learn(data_x[i:i+size],data_y[i:i+size],LR,n,WD)#seems decent enough
                self.batch_num += 1
            self.epoch += 1
        
    def test(self,test_x,test_y):#returns the number of digits correctly identified by the network and the total number of digits tested
        #use zip or for i in test_x?
        #ill try zip first
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
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)#maybe axis 0