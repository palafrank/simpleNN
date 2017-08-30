import sys
import getopt
import numpy as np
import scipy
from scipy import ndimage
from os import listdir

def Read_File(fname):
    a = ndimage.imread(fname, flatten=False)
    b = scipy.misc.imresize(a, size=(64, 64))
    #Normalize data. 255 beecause max pixel value is 255
    b = b/255
    c = b.reshape(64*64*3, 1)
    return c

def collect_data(dirname):
    files = listdir(dirname)
    Y = np.empty((1, len(files)))
    X = np.empty((64*64*3,0))

    i = 0
    for f in files:
        a = Read_File(dirname + "/" + f)
        if f.find("cat") == -1:
            Y[0][i] = 0
        else:
            Y[0][i] = 1
        X = np.hstack((X, a))

        i = i+1
    return X, Y



# Initialize the parameters
# lists_dims["n_x=l0", "l1", "l2" ....]
def initialize_parameters(lists_dims):
    parameters = []
    for i in range(1, len(lists_dims)):
        parameters.append((np.random.randn(lists_dims[i], lists_dims[i-1]) * 0.01, np.zeros((lists_dims[i], 1))))
    return parameters

#Activation Functions
def sigmoid(Z, derivative):
    s = (1/(1+np.exp(-Z)))
    if derivative == True:
        return s*(1-s)
    return s

def relu(Z, derivative):
    for i in range(len(Z)):
        for j in range(len(Z[i])):
            if Z[i][j] > 0:
                if derivative == True:
                    Z[i][j] = 1
                else:
                    pass
            else:
                Z[i][j] = 0
    #print("RELU: ", Z)
    return Z

def forward_activation(Z, func):
    if func == "sigmoid":
        return sigmoid(Z, False)
    elif func == "relu":
        return relu(Z, False)
    else:
        assert(0)
    return None

# Calculate the Z and A parameters of forward propagation for each layer
def forward_propagate(X, parameters, N):
    forward_cache = []
    A = X
    AL = A
    for i in range(N):
        A = AL
        W, b = parameters[i]
        Z = np.dot(W, A) + b
        if i == N-1 :
            activation_func = "sigmoid"
        else:
            activation_func = "sigmoid"
        AL = forward_activation(Z, activation_func)
        forward_cache.append((A, Z, W, b))
    #print (AL)
    return AL, forward_cache

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -(np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1-Y), np.log(1-AL))))/Y.shape[1]
    cost = np.squeeze(cost)
    return cost

def back_propagate_linear(dZ, forward_cache):
    A, Z, W, b = forward_cache
    m = A.shape[1]
    dW = (np.dot(dZ, A.T))/m
    assert(W.shape == dW.shape)
    db = (np.sum(dZ, axis=1, keepdims=True))/m
    assert(b.shape == db.shape)
    dA_prev = np.dot(W.T, dZ)
    assert(dA_prev.shape == A.shape)
    return dA_prev, dW, db

def back_propagate_activation(dA, forward_cache, activation_func):
    A, Z, W, b = forward_cache

    if activation_func == "relu":
        dZ = dA*relu(Z, True)
    elif activation_func == "sigmoid":
        dZ = dA*sigmoid(Z, True)
    else:
        assert(0)

    return back_propagate_linear(dZ, forward_cache)


def back_propagate(AL, Y, forward_cache):
    grads = []
    dA = - (np.divide(Y, AL) - np.divide((1-Y), (1-AL)))
    activation_func = "sigmoid"
    for cache in reversed(forward_cache):
        dA_prev, dW, db = back_propagate_activation(dA, cache, activation_func)
        grads.append((dW, db))
        dA = dA_prev
        activation_func = "sigmoid"
    return grads

def update_parameters(parameters, grads, learning_rate):
    new_params = []
    j=len(parameters)-1
    for i in range(len(parameters)):
        W, b = parameters[i]
        dW, db = grads[j]
        j = j-1
        W = W - (learning_rate * dW)
        b = b - (learning_rate * db)
        new_params.append((W, b))
    return new_params

def calculate_success(Y, AL):
    p = np.around(AL)
    for i in range(len(p[0])):
        if(p[0][i] != Y[0][i]):
            p[0][i] = 0
        else:
            p[0][i] = 1
    return np.squeeze(np.sum(p, axis=1, keepdims=1)/len(p[0]))

def train_model(X, Y, parameters, learning_rate, num_iterations):
    N = len(parameters)
    print ("Num layers:" + str(N))
    for i in range(num_iterations):
        AL, forward_cache = forward_propagate(X, parameters, N)
        c = compute_cost(AL, Y)
        if i % 100 == 0:
            print("Cost at iteration " + str(i) + " : ", str(c))
        grads = back_propagate(AL, Y, forward_cache)
        #print (grads)
        parameters = update_parameters(parameters, grads, learning_rate)
    print ("Trained model success rate: " +  str(calculate_success(Y, AL) *100) + "%")
    return parameters

def test_model(X, Y, parameters):
    m = X.shape[1]
    N = len(parameters)
    AL, forward_cache = forward_propagate(X, parameters, N)
    return calculate_success(Y, AL)
    
def print_help():
    print ("classifier.py -l <learn_dir> -t <test_dir> -r <learn_rate> -i <num iterations> -n \"<comma separated num nodes in each layer>\"")
def main(argv):
    np.random.seed(1)
    learn_dir = "./images"
    test_dir = "./test"
    learning_rate = 0.01
    num_iterations = 4000
    dims = [4, 1]

    try:
        opts, args = getopt.getopt(argv, "hl:t:r:i:n:",["help", "learndir=", "testdir=", "learnrate=", "iters=", "net="])
    except getopt.GetoptError:
        print_help()
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print_help()
            sys.exit(2)
        elif opt == "-l":
            learn_dir = arg
        elif opt == "-t":
            test_dir == arg
        elif opt == "-r":
            learning_rate = float(arg)
        elif opt == "-i":
            num_iterations = int(arg)
        elif opt == "-n":
            dims_str = arg.split(",")
            dims = []
            for num in dims_str:
                dims.append(int(num))
        else:
            print_help()
            sys.exit(2)
    XL, YL = collect_data(learn_dir)
    XT, YT = collect_data(test_dir)
    #Starting Hyperparameters
    lists_dims = [XL.shape[0]]
    for num in dims:
        lists_dims.append(num)
    print(lists_dims)
    #Train the model
    parameters = initialize_parameters(lists_dims)
    parameters = train_model(XL, YL, parameters, learning_rate, num_iterations)
    # check out the success rate with a test run
    success_rate = test_model(XT, YT, parameters)
    print ("Success rate: " + str(success_rate*100) + "%")

if __name__ == "__main__" :
    main(sys.argv[1:])
