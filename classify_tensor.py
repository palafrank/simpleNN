import tensorflow as tf
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

def initialize_parameters(input_layer, list_dims):
    parameters = []
    n_a = input_layer
    for dim in range(len(list_dims)):
        W = tf.get_variable("W"+str(dim), dtype=tf.float64, shape=[list_dims[dim], n_a], initializer =tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("b"+str(dim), dtype=tf.float64, shape=[list_dims[dim], 1], initializer=tf.zeros_initializer)
        parameters.append((W, b))
        n_a = list_dims[dim]
    return parameters

def forward_propagation(X, parameters, layers):
    A = X
    for i in range(layers):
        W, b = parameters[i]
        Z = tf.add(tf.matmul(W, A), b)
        #A = tf.nn.relu(Z, name="Activation"+str(i+1))
        A = tf.sigmoid(Z)
    return Z

def regularize_cost(loss, lamba, parameters):
    W, b = parameters[len(parameters)-1]
    loss = tf.reduce_mean(loss + lamba * tf.nn.l2_loss(W))
    return loss

def regularize_weights(lamba, parameters, m):
    for i in range(len(parameters)):
        W, b = parameters[i]
        W = tf.subtract(W, tf.divide(0.01 * lamba,m) * W)
        parameters[i] = (W, b)
    return parameters



tf.reset_default_graph()
tf.set_random_seed(1)
# Setup the Neural network layers
list_dims = [4, 1]
# Collect the training set
X, Y = collect_data("./images")

# Initialize the weights for every layer
parameters = initialize_parameters(X.shape[0], list_dims)

# Forward propagate
ZL = forward_propagation(X, parameters, len(list_dims))

#Back propagate and reduce cost
AL = tf.nn.sigmoid_cross_entropy_with_logits(labels = Y, logits = ZL)
#cost = regularize_cost(tf.reduce_mean(AL), 0.1, parameters)
cost = tf.reduce_mean(AL)

#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01, name="Optimizer")
optimizer = tf.train.MomentumOptimizer(learning_rate = 0.01, momentum=0.9)
train = optimizer.minimize(cost)


# Setup session and initialize tensors
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Run the training iterations
costs = []
for i in range(6000):
    _, new_cost, _ = sess.run([AL,cost, train])
    #regularize_weights(0.1, parameters, X.shape[1])
    if (i % 1000) == 0:
        print("Training iteration " + str(i), new_cost)
    costs.append(new_cost)
print("Final cost = ", costs[-1])
#print(sess.run(loss))

#merged = tf.summary.merge_all()
#my_writer = tf.summary.FileWriter("./", sess.graph)
parameters = sess.run(parameters)

X_test, Y_test = collect_data("./test")
predict_ZL = forward_propagation(X_test, parameters, len(list_dims))
predict_AL = tf.sigmoid(predict_ZL)
print("Test data prediction:", sess.run(predict_AL))

sess.close()
