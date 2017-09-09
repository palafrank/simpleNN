import sys
import getopt
import numpy as np
import scipy
from scipy import ndimage
from os import listdir
import matplotlib.pyplot as plt

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

def initialize_hyperparams():
    hyperparams = {}
    hyperparams["learning_rate"] = 0.01 # Learning rate of gradient descent
    hyperparams["num_iterations"] = 4000 # Number of iterations of propagation
    hyperparams["dims"] = [4, 1] # Number of nodes in each layer of NN
    hyperparams["lamba"] = 0.01 # Regularization param lambda
    hyperparams["beta1"] = 0.9 # Exponential Weighted average param
    hyperparams["beta2"] = 0.999 # RMSProp param
    hyperparams["epsilon"] = 10 ** -8 # Adams optimization zero correction
    return hyperparams

# Initialize the parameters
# lists_dims["n_x=l0", "l1", "l2" ....]
def initialize_parameters(lists_dims):
    parameters = []
    for i in range(1, len(lists_dims)):
        # 'He' initialization for random weights
        W = np.random.randn(lists_dims[i], lists_dims[i-1]) * np.sqrt(np.divide(2, lists_dims[i-1]))
        Vw = np.zeros(W.shape)
        Sw = np.zeros(W.shape)
        b = np.zeros((lists_dims[i], 1))
        Vb = np.zeros(b.shape)
        Sb = np.zeros(b.shape)
        parameters.append((W, b, Vw, Vb, Sw, Sb))
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
        W, b, Vw, Vb, Sw, Sb = parameters[i]
        Z = np.dot(W, A) + b
        if i == N-1 :
            activation_func = "sigmoid"
        else:
            activation_func = "relu"
        AL = forward_activation(Z, activation_func)
        forward_cache.append((A, Z, W, b))
    #print (AL)
    return AL, forward_cache

def regularize_cost(cost, m, lamba, parameters):
    if lamba == 0:
        return cost
    W, b, Vw, Vb, Sw, Sb = parameters[len(parameters)-1]
    n = np.linalg.norm(W)
    normalize = np.divide(lamba, 2*m)*n
    cost = cost + normalize
    return cost

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
        activation_func = "relu"
    return grads

def regularize_weights(W, m, learning_rate, lamba):
    if lamba == 0:
        return W
    W = W - np.multiply(np.divide(np.multiply(learning_rate, lamba), m), W)
    return W

# A combination of Exponential Weighted Average and RMSProp with bias correction
# To be used with mini-batches

def momentum(dW, db, Vw, Vb, hyperparams, layer):
    beta1 = hyperparams["beta1"]
    Vw = np.multiply(beta1, Vw) + np.multiply((1-beta1), dW)
    Vw_corrected = np.divide(Vw, (1-np.power(beta1, layer)))
    Vb = np.multiply(beta1, Vb) + np.multiply((1-beta1), db)
    Vb_corrected = np.divide(Vb, (1-np.power(beta1, layer)))
    return Vw_corrected, Vb_corrected, Vw, Vb

def rms_prop(dW, db, Sw, Sb, hyperparams, layer):
    beta2 = hyperparams["beta2"]
    epsilon = hyperparams["epsilon"]
    Sw = np.multiply(beta2, Sw) + np.multiply((1-beta2), np.power(dW,2))
    Sw_corrected = np.divide(Sw, (1-np.power(beta2, layer)))
    Sb = np.multiply(beta2, Sb) + np.multiply((1-beta2), np.power(db, 2))
    Sb_corrected = np.divide(Sb, (1-np.power(beta2, layer)))
    dW_optimized = np.divide(dW, (np.sqrt(Sw_corrected + epsilon)))
    db_optimized = np.divide(db, (np.sqrt(Sb_corrected + epsilon)))
    return dW_optimized, db_optimized, Sw, Sb

def adams_optimization(dW, db, Vw, Vb, Sw, Sb, hyperparams, layer):
    beta1 = hyperparams["beta1"]
    beta2 = hyperparams["beta2"]
    epsilon = hyperparams["epsilon"]
    Vw = np.multiply(beta1, Vw) + np.multiply((1-beta1), dW)
    Vw_corrected = np.divide(Vw, (1-np.power(beta1, layer)))
    Sw = np.multiply(beta2, Sw) + np.multiply((1-beta2), np.power(dW, 2))
    Sw_corrected = np.divide(Sw, (1-np.power(beta2, layer)))
    Vb = np.multiply(beta1, Vb) + np.multiply((1-beta1), db)
    Vb_corrected = np.divide(Vb, (1-np.power(beta1, layer)))
    Sb = np.multiply(beta2, Sb) + np.multiply((1-beta2), np.power(db, 2))
    Sb_corrected = np.divide(Sb, (1-np.power(beta2, layer)))
    dW_optimized = np.divide(Vw_corrected, (np.sqrt(Sw_corrected) + epsilon))
    db_optimized = np.divide(Vb_corrected, (np.sqrt(Sb_corrected) + epsilon))
    return dW_optimized, db_optimized, Vw, Vb, Sw, Sb

def update_parameters(m, parameters, grads, hyperparams):
    new_params = []
    learning_rate = hyperparams["learning_rate"]
    j=len(parameters)-1
    for i in range(len(parameters)):
        W, b, Vw, Vb, Sw, Sb = parameters[i]
        dW, db = grads[j]
        #dW, db, Vw, Vb, Sw, Sb = adams_optimization(dW, db, Vw, Vb, Sw, Sb, hyperparams, j+1)
        dW, db, Vw, Vb = momentum(dW, db, Vw, Vb, hyperparams, j+1)
        #dW, db, Sw, Sb = rms_prop(dW, db, Sw, Sb, hyperparams, j+1)
        j = j-1
        W = W - (learning_rate * dW)
        W = regularize_weights(W, m, learning_rate, hyperparams["lamba"])
        b = b - (learning_rate * db)
        new_params.append((W, b, Vw, Vb, Sw, Sb))
    return new_params

def calculate_success(Y, AL):
    p = np.around(AL)
    for i in range(len(p[0])):
        if(p[0][i] != Y[0][i]):
            p[0][i] = 0
        else:
            p[0][i] = 1
    return np.squeeze(np.sum(p, axis=1, keepdims=1)/len(p[0]))

def train_model(X, Y, parameters, hyperparams):
    num_iterations = hyperparams["num_iterations"]
    N = len(parameters)
    m = X.shape[1]
    cost = []
    print ("Num layers:" + str(N))
    for i in range(num_iterations):
        AL, forward_cache = forward_propagate(X, parameters, N)
        c = compute_cost(AL, Y)
        c = regularize_cost(c, m, hyperparams["lamba"], parameters)
        cost.append(c)
        if i % 100 == 0:
            print("Cost at iteration " + str(i) + " : ", str(c))
        grads = back_propagate(AL, Y, forward_cache)
        #print (grads)
        parameters = update_parameters(m, parameters, grads, hyperparams)
    print ("Trained model success rate: " +  str(calculate_success(Y, AL) *100) + "%")
    return parameters, cost

def test_model(X, Y, parameters):
    m = X.shape[1]
    N = len(parameters)
    AL, forward_cache = forward_propagate(X, parameters, N)
    return calculate_success(Y, AL)

def print_help():
    print ("classifier.py -l <learn_dir> -t <test_dir> -r <learn_rate> -i <num iterations> -n \"<comma separated num nodes in each layer>\"")

def plot_cost_gradient(cost):
    plt.plot(cost)
    plt.ylabel("Cost")
    plt.xlabel("Per 100 iterations")
    plt.title("Cost gradient")
    plt.show()

def main(argv):
    np.random.seed(1)
    learn_dir = "./images"
    test_dir = "./test"
    hyperparams = initialize_hyperparams()

    dims = hyperparams["dims"]

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
    parameters, cost = train_model(XL, YL, parameters, hyperparams)
    # check out the success rate with a test run
    success_rate = test_model(XT, YT, parameters)
    print ("Test Success rate: " + str(success_rate*100) + "%")
    plot_cost_gradient(cost)

if __name__ == "__main__" :
    main(sys.argv[1:])
