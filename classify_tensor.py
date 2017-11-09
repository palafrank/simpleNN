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
'''
def shuffle_data(X, Y, batch_size=250):
    X_dims = X.shape
    Y_dims = Y.shape
    #print(X_dims, Y_dims)
    num_mini = int(X.shape[1]/batch_size)
    seq = [i for i in range(X_dims[1])]
    np.random.shuffle(seq)
    minibatches = []
    seq_index = 0
    for i in range(num_mini):
        mb_x = np.empty((X.shape[0],0))
        mb_y = np.empty((1, 0))
        for j in range(batch_size):
            mb_x = np.hstack((mb_x, X[:,seq[seq_index]:seq[seq_index]+1]))
            mb_y = np.hstack((mb_y, Y[:,seq[seq_index]:seq[seq_index]+1]))
            seq_index = seq_index + 1
        #print(mb_x.shape, mb_y.shape)
        minibatches.append((mb_x, mb_y))
    return minibatches

'''
def shuffle_data(X, Y):
    seq = [i for i in range(X.shape[1])]
    np.random.shuffle(seq)
    minibatches = []
    for i, data in enumerate(seq):
        X[:, i], X[:, data] = X[:, data], X[:, i]
        Y[:, i], Y[:, data] = Y[:, data], Y[:, i]
    minibatches.append((X,Y))
    return minibatches

tf.reset_default_graph()
tf.set_random_seed(1)
X, Y = collect_data("./images")

X_test, Y_test = collect_data("./test")
x = tf.placeholder(tf.float32, shape=[X.shape[0], None], name="Input")
y = tf.placeholder(tf.float32, shape=[1, None], name="Label")

#W0 = tf.Variable(tf.zeros([10, X.shape[0]]), name="W0")
#b0 = tf.Variable(tf.zeros([10,1]), name="b0")
W0 = tf.get_variable("W0", dtype=tf.float32, shape=[10, X.shape[0]], initializer =tf.contrib.layers.xavier_initializer())
b0 = tf.get_variable("b0", dtype=tf.float32, shape=[10, 1], initializer=tf.zeros_initializer)
Z0 = tf.add(tf.matmul(W0, x),b0)
A0 = tf.nn.relu(Z0)
W1 = tf.get_variable("W1", dtype=tf.float32, shape=[1, 10], initializer =tf.contrib.layers.xavier_initializer())
b1 = tf.get_variable("b1", dtype=tf.float32, shape=[1, 1], initializer=tf.zeros_initializer)
Z1 = tf.add(tf.matmul(W1, A0), b1)
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = y, logits = Z1, name="Activation"))

predict = tf.sigmoid(Z1)
success_rate = tf.multiply(tf.reduce_mean(tf.cast(tf.equal(tf.round(predict), y), tf.float32)), 100)

train_step = tf.train.AdamOptimizer().minimize(cost)
#train_step = tf.train.MomentumOptimizer(learning_rate = 0.01, momentum=0.9, name="Optimizer").minimize(A1)
#train_step = tf.train.AdamOptimizer(name="Optimizer").minimize(A1)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
writer = tf.summary.FileWriter("/tmp/tensor_classify")
writer.add_graph(sess.graph)
num_epochs = 2000
costs = []
np.random.seed(1)
for i in range(num_epochs):
    mini_batches = shuffle_data(X, Y)
    for data in mini_batches:
        data_x, data_y = data
        #print(data_x.shape, data_y.shape)
        mycost, _ = sess.run([cost, train_step], feed_dict={x:data_x, y:data_y})
        costs.append(mycost)
    if i % 100 == 0:
        cal_cost = sess.run(cost, feed_dict={x:X, y:Y})
        print("Cost:", cal_cost)
        print(sess.run(success_rate, feed_dict={x:X, y:Y}), "%")

print(sess.run(success_rate, feed_dict={x:X_test, y:Y_test}), "%")

sess.close()
