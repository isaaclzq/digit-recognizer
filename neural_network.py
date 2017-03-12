from mnist import MNIST
import numpy as np
import random
import datetime

NUM_CLASSES = 10


"""
Change this code however you want.
"""
def load_dataset():
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, _ = map(np.array, mndata.load_testing())
    
    
    y_train = one_hot(labels_train)
    xtest_normal = normalize(X_test)
    xtrain_normal = normalize(X_train)
    
    return X_train, xtrain_normal, labels_train, y_train, X_test, xtest_normal

def normalize(matrix):
    ans = np.zeros(matrix.shape)
    for x in range(ans.shape[0]):
        ans[x] = matrix[x] - np.average(matrix[x])
        ans[x] = ans[x] / 255.0
    return ans

def one_hot(labels_train):
    '''Convert categorical labels 0,1,2,....9 to standard basis vectors in R^{10} '''
    return np.array([np.eye(NUM_CLASSES)[num] for num in labels_train])

X_train, xtrain_normal, labels_train, y_train, X_test, xtest_normal = load_dataset()

def relu_activation(matrix):
    relu_func = np.vectorize(lambda x: x if x > 0 else 0, otypes=[np.float])
    return relu_func(matrix)

def relu_derivative(matrix):
    relu_deri = np.vectorize(lambda x: 1 if x > 0 else 0, otypes=[np.float])
    return relu_deri(matrix)

def sigmoid_activation(matrix):
    return 1.0 / (1.0 + np.exp(-matrix))

def sigmoid_derivative(matrix):
    return matrix * (1 - matrix)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def softmax_activation(x):
    return np.apply_along_axis(softmax, 1, x)    

def softmax_derivative(matrix):
    return matrix * (1 - matrix)

def entropy(x, y):
    return -np.inner(x, y)

def cross_entropy(x, y):
    ans = 0
    row, column = x.shape
    for i in range(column):
        ans += entropy(x.T[i], y.T[i])
    return -ans

def forward_pass(x, w, v, input_layer=784, hidden_layer=200):
    s_1 = np.append(x, 1).reshape(1, input_layer + 1).dot(w) 
    y_i = relu_activation(s_1)
    s_2 = np.append(y_i, 1).reshape(1, hidden_layer + 1).dot(v) 
    out = softmax_activation(s_2)
    return s_1, y_i, s_2, out

learning_rate = 0.01
num_iteration = 300000
def train(X=xtrain_normal, Y=y_train, learning_rate=learning_rate, iterations=num_iteration, lr_change=False, record=None):
    num_sample, input_layer = X.shape
    hidden_layer = 200
    output_layer = 10
 
    w = np.random.normal(0,0.2, (input_layer+1)*hidden_layer).reshape(input_layer+1, hidden_layer)
    v = np.random.normal(0,0.2, (hidden_layer+1)*output_layer).reshape(hidden_layer+1, output_layer)
    
    i = 0
    while i < iterations:
 
        index = random.randrange(len(Y))
        x = X[index]
        y = Y[index]

        # forward pass for sgd
        s_1, y_i, s_2, out = forward_pass(x,w,v,input_layer,hidden_layer)

        #backprop calculation
        delta_2 = out - y
        dJdv_ij = np.append(y_i, 1).reshape(hidden_layer + 1, 1).dot(delta_2)

        delta_1 = delta_2.dot(v[:-1,:].T) * relu_derivative(s_1)         
        dJdw_ij = np.append(x, 1).reshape(input_layer + 1, 1).dot(delta_1) 
        
        if lr_change:
            learning_rate = learning_rate/i
        
        w = w - learning_rate * dJdw_ij
        v = v - learning_rate * dJdv_ij

        if (i+1) % 50000 == 0:
            learning_rate = learning_rate / 10.0

        if record != None and i % 1000 == 0:
            print ("Finished " + str(i) + " iterations.")
 
        i += 1
    return w, v

def predict(w, v, X):
    num_sample, input_layer = X.shape

    # forward pass in batch
    s_1 = np.concatenate((X, np.ones((num_sample, 1))), axis=1).dot(w) 
    y_i = relu_activation(s_1)
    s_2 = np.concatenate((y_i, np.ones((num_sample, 1))), axis=1).dot(v) 
    out = softmax_activation(s_2) 
    return np.argmax(out, axis=1)


err_t = []
start = datetime.datetime.now()
w_ij, v_ij = train(xtrain_normal, y_train, record=err_t)
end = datetime.datetime.now()
print ("training time: ")
print (end - start)
 
start = datetime.datetime.now()
predictions = predict(w_ij, v_ij, xtrain_normal)
end = datetime.datetime.now()
print ("[training result] predict time: ")
print (end - start)
print (np.true_divide((predictions == labels_train).sum(), len(labels_train)))


