import numpy as np
import random
import matplotlib.pyplot as plt

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
# input dataset
training_data = [
    (np.array([0,0,-1]), 0, 0),
    (np.array([0,1,-1]), 1, 1),
    (np.array([1,0,-1]), 1, 2),
    (np.array([1,1,-1]), 0, 3),
]
    
#set learning rate
alpha = 0.5
epoch = 0

# output dataset 
outputs = np.zeros((4,5)) 
outputs[0][0] = training_data[0][0][0]
outputs[0][1] = training_data[0][0][1]
outputs[0][2] = training_data[0][1]
outputs[1][0] = training_data[1][0][0]
outputs[1][1] = training_data[1][0][1]
outputs[1][2] = training_data[1][1]
outputs[2][0] = training_data[2][0][0]
outputs[2][1] = training_data[2][0][1]
outputs[2][2] = training_data[2][1]
outputs[3][0] = training_data[3][0][0]
outputs[3][1] = training_data[3][0][1]
outputs[3][2] = training_data[3][1]
outputs[0][4] = 1
outputs[1][4] = 1
outputs[2][4] = 1
outputs[3][4] = 1
e_squared = (outputs[0][4]**2) + (outputs[1][4]**2) +(outputs[2][4]**2) + (outputs[3][4]**2)
e_track = []
e_track.append(e_squared)

# initialize weights randomly with mean 0
w13 = (random.uniform(-1.2,1.2))
w14 = (random.uniform(-1.2,1.2))
w23 = (random.uniform(-1.2,1.2))
w24 = (random.uniform(-1.2,1.2))
th3 = (random.uniform(-1.2,1.2))
th4 = (random.uniform(-1.2,1.2))
syn0 = np.zeros((3,2))
syn0[0][0] = w13
syn0[0][1] = w14
syn0[1][0] = w23
syn0[1][1] = w24
syn0[2][0] = th3
syn0[2][1] = th4    
syn0 = np.matrix(syn0)
syn0_update = syn0
#print(syn0)        
#w13 w14
#w23 w24
#th3 th4
          
syn1 = np.zeros((1,3))
w35 = (random.uniform(-1.2,1.2))
w45 = (random.uniform(-1.2,1.2))
th5 = (random.uniform(-1.2,1.2))
syn1[0][0] = w35
syn1[0][1] = w45
syn1[0][2] = th5
syn1=np.matrix(syn1)   
syn1_update = syn1
#w35 w45 th5

 #begin iterating here
 #forward propagation
while(e_squared > .001):
    l0,y,data_id=random.choice(training_data)
    l0=l0.reshape(1,3)
    l1 = nonlin(l0*syn0_update)
    a=np.array([-1])
    l1=np.array(l1)
    l1=np.concatenate((l1[0],a))
    l2=nonlin(l1*syn1_update.T)
    e = y - l2
    outputs[data_id][3] = l2
    outputs[data_id][4] = e
    
    # back propagation of error
    y3 = l1[0]
    y4 = l1[1]
    egrad5 = l2* (1-l2) * e
    dw35 = alpha * y3 * egrad5
    dw45 = alpha * y4 * egrad5
    dth5 = alpha * a * egrad5
    egrad3 = y3 * (1-y3) * egrad5 * w35
    egrad4 = y4 * (1-y4) * egrad5 * w45
    
    # determine weight corrections
    dw13 = alpha * l0[0][0] * egrad3
    dw23 = alpha * l0[0][1] * egrad3
    dth3 = alpha * l0[0][2] * egrad3
    dw14 = alpha * l0[0][0] * egrad4
    dw24 = alpha * l0[0][1] * egrad4
    dth4 = alpha * l0[0][2] * egrad4
    
    w13 = w13 + dw13
    w23 = w23 + dw23
    th3 = th3 + dth3 
    w14 = w14 + dw14
    w24 = w24 + dw24
    th4 = th4 + dth4
    w35 = w35 + dw35
    w45 = w45 + dw45
    th5 = th5 + dth5
    
    syn0_update = np.zeros((3,2))
    syn0_update[0][0] = w13
    syn0_update[0][1] = w14
    syn0_update[1][0] = w23
    syn0_update[1][1] = w24
    syn0_update[2][0] = th3
    syn0_update[2][1] = th4    
    syn0_update = np.matrix(syn0_update)
    
    syn1_update = np.zeros((1,3))
    syn1_update[0][0] = w35
    syn1_update[0][1] = w45
    syn1_update[0][2] = th5
    syn1_update=np.matrix(syn1_update)
    
    e_squared = (outputs[0][4]**2) + (outputs[1][4]**2) +(outputs[2][4]**2) + (outputs[3][4]**2)
    e_track.append(e_squared)
    epoch = epoch + 1
    if epoch%10000 == 0:
        print(e_squared)
    
print("Input 1, Input 2, Expected Output, Actual Output, Error")
print(outputs)
print("Initial Weights")
print(syn0)
print(syn1)
print("Final Weights")
print(syn0_update)
print(syn1_update)
print("Number of epochs")
print(epoch)

plt.semilogy(e_track)
plt.title("Squared Error per Epoch")
plt.ylabel("Sum of the squared errors")
plt.xlabel("Epoch number")
plt.show()