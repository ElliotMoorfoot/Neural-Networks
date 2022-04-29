
import Standard_Network

import numpy as np
import math
from keras.datasets import mnist
import matplotlib.pyplot as plt




#loading the dataset
(train_x, train_y), (test_x, test_y) = mnist.load_data()

# train_x is the 28x28 pixels and train_y is the number (desired output)
train_x = (1/255)*train_x #mapping the data from (0,255) --> (0,1)
test_x = (1/255)*test_x # i may have done this somewhere in network i dont remember

'''
data augmentation
'''
#image = train_x[5] 
#fig = plt.figure
#plt.imshow(image, cmap='gray_r')
#plt.show()

'''
translations
'''
left = np.zeros((60000,28,28))
left[:,:,:-1] = train_x[:,:,1:]
up = np.zeros((60000,28,28))
up[:,:-1,:] = train_x[:,1:,:]
right = np.zeros((60000,28,28))
right[:,:,1:] = train_x[:,:,:-1]
down = np.zeros((60000,28,28))
down[:,1:,:] = train_x[:,:-1,:]



#plt.imshow(up[5], cmap='gray_r')
#plt.show()

#images = np.zeros((5,28,28))
images = np.array( ((train_x[5]),(up[5]),(down[5]),(left[5]),(right[5]) ))
labels = np.array(('Original','Up','Down','Left','Right'))
num_row = 1
num_col = 5
#fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
#for i in range(5):
    #ax = axes[i//num_col, i%num_col]
 #   ax = axes[i%num_col]
  #  ax.imshow(images[i], cmap='gray')
   # ax.set_title('{}'.format(labels[i]))
#plt.tight_layout()
#plt.show()



train_y = np.reshape(train_y,(60000,1)) #this might save time if it works
test_y = np.reshape(test_y,(10000,1))

up = np.reshape(up,(60000,784,1))
down = np.reshape(down,(60000,784,1))
left = np.reshape(left,(60000,784,1))
right = np.reshape(right,(60000,784,1))
train_x = np.reshape(train_x,(60000,784,1)) #this might save time if it works
test_x = np.reshape(test_x,(10000,784,1))


net = Standard_Network.Network([784,100,10])

print(net.test(test_x,test_y))
print(net.test(train_x,train_y))


def plot_acc(epoch):
    cost = []
    cost_test = []
    num = [i for i in range(1,epoch)]
    for i in range(epoch):
        cost_test.append(net.test(test_x,test_y)[0]/100)
        cost.append(net.test(train_x,train_y)[0]/600)
        net.train(train_x,train_y,0.001,10,1,0)
        #net.train(up,train_y,0.1,10,1,0)
        #net.train(left,train_y,0.1,10,1,0)
        #net.train(right,train_y,0.1,10,1,0)
        #net.train(down,train_y,0.1,10,1,0)
        print(i)
    plt.plot(num,cost[1:],'-r',label='Training set')
    plt.plot(num,cost_test[1:],'-b',label='Test set')
    plt.legend(loc="upper left")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy(%)")
    plt.title("Learning Rate Decay")
    plt.show()
#plot_acc(20)
    
def quad(a,y):
    val = 0
    for i in range(10):
        if i == y:
            val += (a[i,0]-1)**2
        else:
            val += a[i,0]**2
    return val

def cross(a,y):
    val = 0
    for i in range(10):
        if i == y:
            val -= np.log(a[i,0])
        else:
            val -= np.log(1-a[i,0])
    return val

def cross2(a,y):
    Y = np.zeros((10,1))
    Y[y] = 1
    return np.sum(Y*np.log(a) + (1-Y)*np.log(1-a))
 
def plot_cost(epoch):
    cost = []
    cost_test = []
    num = [i for i in range(1,epoch)]
    for i in range(epoch):
        tot = 0
        for x,y in zip(train_x,train_y):
            tot += quad(net.output(x),y)
        cost.append(tot/120000)
        tot = 0
        for x,y in zip(test_x,test_y):
            tot += quad(net.output(x),y)
        cost_test.append(tot/20000)
        net.train(train_x,train_y,0.01,10,1,0,WD=0.5)
       # net.train(up,train_y,0.1,10,1,0)
       # net.train(left,train_y,0.1,10,1,0)
       # net.train(right,train_y,0.1,10,1,0)
        #net.train(down,train_y,0.1,10,1,0)
        print(i)
    plt.plot(num,cost[1:],'-r',label='Training set')
    plt.plot(num,cost_test[1:],'-b',label='Test set')
    plt.legend(loc="upper left")
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.title("Weight Decay = 0.5")
    plt.show()
plot_cost(50)
    
    
    
    
#    net.train(left,train_y,0.1,32,1,0.5)
 #   net.train(up,train_y,0.1,32,1,0.5)
  #  net.train(down,train_y,0.1,32,1,0.5)
   # net.train(right,train_y,0.1,32,1,0.5)
    #print(net.test(test_x,test_y))
    #print(net.test(train_x,train_y))

    
'''i think we can train the net over the same training data but shuffle it each time
add this to train function for quality of life changes. add a shuffle func too'''


'''network3: 9666 max correct with LR = 0.01, 9674 with LR = 0.001, 9735 with decay'''

'''training in steps'''
#m = 100 # steps
#for i in range(0,60000,m):
 #   net.train(train_X[i:i+m],train_y[i:i+m],0.2,10)
  #  print (net.test(test_X,test_y))
  
'''
cross-entropy function
softmax output layor and log cost (for deep networks)
early stopping
activation data for early stopping and setting hyper peramitors without overfitting them to test data "hold out" method
weight decay/L2 regularization - regularized cross-entropy func
L1 regularization
drop out
artifically increasing the training data
im gonna use L2 over L1 maybe look into why?
weight initislation (link to more in depth ideas)
might not need weight down scaled with L2 method (weight decay will do it on first run through)
'''
