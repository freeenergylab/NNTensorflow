#!/usr/bin/env python

import tensorflow as tf
import symmetry_functions
import numpy as np
import matplotlib.pyplot as plt

#===============================================================================
# Number of QM region atoms
#===============================================================================
# Note: dependent on your system
QM_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

#===============================================================================
# Symmetry Function 1: Radial Function
#===============================================================================
# in unit of 1/(A**2) and Bohr_to_A = 0.529177249
eta = {'C': 0.714, 'C1': 0.714, 'C2': 0.714, 'C3': 0.714, 'C4': 0.714, \
       'O': 2.857, 'H': 1.428, 'H1': 1.428, 'H2': 1.428, 'H3': 1.428, \
       'H4': 1.428, 'H5': 1.428, 'H6': 1.428, 'H7': 1.428}
G1 = symmetry_functions.RadialFunction(R_s = 0.0, R_c = 6.0, eta = eta, 
                    prmtop = '../training_data/ligands.parm7', 
                    mdcrd = '../training_data/md.crd', 
                    nframe = 1000, QM_index = QM_index)

#===============================================================================
# Symmetry Function 2: Angular Function
#===============================================================================
ksi = {'C': 0.2, 'C1': 0.2, 'C2': 0.2, 'C3': 0.2, 'C4': 0.2, 'O': 0.8, \
       'H': 0.4, 'H1': 0.4, 'H2': 0.4, 'H3': 0.4, 'H4': 0.4, 'H5': 0.4, \
       'H6': 0.4, 'H7': 0.4}
# Note: lmda = 1.0 or -1.0
lmda = 1.0
G2 = symmetry_functions.AngularFunction(R_c = 6.0, lmda = lmda, eta = eta, ksi = ksi, 
                     prmtop = '../training_data/ligands.parm7', 
                     mdcrd = '../training_data/md.crd', 
                     nframe = 1000, QM_index = QM_index)

#===============================================================================
# Create x, y (np.array)
#===============================================================================
infile = open('../training_data/EB3LYP-EPM6.dat', 'r')
lines = infile.readlines()
infile.close()

nframes = len(lines)
Ediff = np.zeros([nframes], np.float32)
for k in range(nframes):
    line = lines[k]
    tokens = line.split()
    Ediff[k] = float(tokens[1])
y_data = Ediff
#print(y_data)

nqm = len(QM_index)
x_data = np.zeros([nframes, nqm*2], np.float32)
for k in range(nframes):
    n = 0
    for i in range(nqm):
        x_data[k, n+0] = G1[k, i]
        x_data[k, n+1] = G2[k, i]
        n += 2
#print(x_data)
#print(x_data[:,0:2])
#print(x_data.shape)
#print(len(x_data))

#===============================================================================
# Construct Neural Network for QM/MM
#===============================================================================
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

ys = tf.placeholder(shape=[None, 1], dtype=tf.float32)

#===============================================================================
xs_0 = tf.placeholder(shape=[None, 2], dtype=tf.float32)
l_01 = add_layer(xs_0, 2, 20, activation_function=tf.nn.tanh)
l_02 = add_layer(l_01, 20, 10, activation_function=tf.nn.tanh)
pred_0 = add_layer(l_02, 10, 1, activation_function=tf.nn.tanh)

xs_1 = tf.placeholder(shape=[None, 2], dtype=tf.float32)
l_11 = add_layer(xs_1, 2, 20, activation_function=tf.nn.tanh)
l_12 = add_layer(l_11, 20, 10, activation_function=tf.nn.tanh)
pred_1 = add_layer(l_12, 10, 1, activation_function=tf.nn.tanh)

xs_2 = tf.placeholder(shape=[None, 2], dtype=tf.float32)
l_21 = add_layer(xs_2, 2, 20, activation_function=tf.nn.tanh)
l_22 = add_layer(l_21, 20, 10, activation_function=tf.nn.tanh)
pred_2 = add_layer(l_22, 10, 1, activation_function=tf.nn.tanh)

xs_3 = tf.placeholder(shape=[None, 2], dtype=tf.float32)
l_31 = add_layer(xs_3, 2, 20, activation_function=tf.nn.tanh)
l_32 = add_layer(l_31, 20, 10, activation_function=tf.nn.tanh)
pred_3 = add_layer(l_32, 10, 1, activation_function=tf.nn.tanh)

xs_4 = tf.placeholder(shape=[None, 2], dtype=tf.float32)
l_41 = add_layer(xs_4, 2, 20, activation_function=tf.nn.tanh)
l_42 = add_layer(l_41, 20, 10, activation_function=tf.nn.tanh)
pred_4 = add_layer(l_42, 10, 1, activation_function=tf.nn.tanh)

xs_5 = tf.placeholder(shape=[None, 2], dtype=tf.float32)
l_51 = add_layer(xs_5, 2, 20, activation_function=tf.nn.tanh)
l_52 = add_layer(l_51, 20, 10, activation_function=tf.nn.tanh)
pred_5 = add_layer(l_52, 10, 1, activation_function=tf.nn.tanh)

xs_6 = tf.placeholder(shape=[None, 2], dtype=tf.float32)
l_61 = add_layer(xs_6, 2, 20, activation_function=tf.nn.tanh)
l_62 = add_layer(l_61, 20, 10, activation_function=tf.nn.tanh)
pred_6 = add_layer(l_62, 10, 1, activation_function=tf.nn.tanh)

xs_7 = tf.placeholder(shape=[None, 2], dtype=tf.float32)
l_71 = add_layer(xs_7, 2, 20, activation_function=tf.nn.tanh)
l_72 = add_layer(l_71, 20, 10, activation_function=tf.nn.tanh)
pred_7 = add_layer(l_72, 10, 1, activation_function=tf.nn.tanh)

xs_8 = tf.placeholder(shape=[None, 2], dtype=tf.float32)
l_81 = add_layer(xs_8, 2, 20, activation_function=tf.nn.tanh)
l_82 = add_layer(l_81, 20, 10, activation_function=tf.nn.tanh)
pred_8 = add_layer(l_82, 10, 1, activation_function=tf.nn.tanh)

xs_9 = tf.placeholder(shape=[None, 2], dtype=tf.float32)
l_91 = add_layer(xs_9, 2, 20, activation_function=tf.nn.tanh)
l_92 = add_layer(l_91, 20, 10, activation_function=tf.nn.tanh)
pred_9 = add_layer(l_92, 10, 1, activation_function=tf.nn.tanh)

xs_10 = tf.placeholder(shape=[None, 2], dtype=tf.float32)
l_101 = add_layer(xs_10, 2, 20, activation_function=tf.nn.tanh)
l_102 = add_layer(l_101, 20, 10, activation_function=tf.nn.tanh)
pred_10 = add_layer(l_102, 10, 1, activation_function=tf.nn.tanh)

xs_11 = tf.placeholder(shape=[None, 2], dtype=tf.float32)
l_111 = add_layer(xs_11, 2, 20, activation_function=tf.nn.tanh)
l_112 = add_layer(l_111, 20, 10, activation_function=tf.nn.tanh)
pred_11 = add_layer(l_112, 10, 1, activation_function=tf.nn.tanh)

xs_12 = tf.placeholder(shape=[None, 2], dtype=tf.float32)
l_121 = add_layer(xs_12, 2, 20, activation_function=tf.nn.tanh)
l_122 = add_layer(l_121, 20, 10, activation_function=tf.nn.tanh)
pred_12 = add_layer(l_122, 10, 1, activation_function=tf.nn.tanh)

xs_13 = tf.placeholder(shape=[None, 2], dtype=tf.float32)
l_131 = add_layer(xs_13, 2, 20, activation_function=tf.nn.tanh)
l_132 = add_layer(l_131, 20, 10, activation_function=tf.nn.tanh)
pred_13 = add_layer(l_132, 10, 1, activation_function=tf.nn.tanh)
#===============================================================================

temp01 = tf.add(pred_0, pred_1)
temp02 = tf.add(temp01, pred_2)
temp03 = tf.add(temp02, pred_3)
temp04 = tf.add(temp03, pred_4)
temp05 = tf.add(temp04, pred_5)
temp06 = tf.add(temp05, pred_6)
temp07 = tf.add(temp06, pred_7)
temp08 = tf.add(temp07, pred_8)
temp09 = tf.add(temp08, pred_9)
temp10 = tf.add(temp09, pred_10)
temp11 = tf.add(temp10, pred_11)
temp12 = tf.add(temp11, pred_12)
prediction = tf.add(temp12, pred_13)

loss = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1])))

#train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
train_step = tf.train.AdamOptimizer(0.1).minimize(loss)

#===============================================================================
# Training and Testing
#===============================================================================

#===============================================================================
# 80% of data for training 
train_indices = np.random.choice(len(x_data), round(len(x_data)*0.8), replace=False)

x_data_train = x_data[train_indices]
y_data_train = y_data[train_indices]

#print(x_data_train)
#print(x_data_train.shape)
#print(x_data_train[:,0:2])
#print(y_data_train)
#print(y_data_train.shape)
#===============================================================================

#===============================================================================
# 20% of data for testing
test_indices = np.array(list(set(range(len(x_data))) - set(train_indices)))

x_data_test = x_data[test_indices]
y_data_test = y_data[test_indices]
#===============================================================================

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

train_loss = []
test_loss = []
for i in range(2000):
    # training
    sess.run(train_step, feed_dict={xs_0: x_data_train[:, 0:2], \
                                    xs_1: x_data_train[:, 2:4], \
                                    xs_2: x_data_train[:, 4:6], \
                                    xs_3: x_data_train[:, 6:8], \
                                    xs_4: x_data_train[:, 8:10], \
                                    xs_5: x_data_train[:, 10:12], \
                                    xs_6: x_data_train[:, 12:14], \
                                    xs_7: x_data_train[:, 14:16], \
                                    xs_8: x_data_train[:, 16:18], \
                                    xs_9: x_data_train[:, 18:20], \
                                    xs_10: x_data_train[:, 20:22], \
                                    xs_11: x_data_train[:, 22:24], \
                                    xs_12: x_data_train[:, 24:26], \
                                    xs_13: x_data_train[:, 26:28], \
                                    ys: y_data_train.reshape([len(y_data_train), 1])})

    train_temp_loss = sess.run(loss, feed_dict={xs_0: x_data_train[:, 0:2], \
                                                xs_1: x_data_train[:, 2:4], \
                                                xs_2: x_data_train[:, 4:6], \
                                                xs_3: x_data_train[:, 6:8], \
                                                xs_4: x_data_train[:, 8:10], \
                                                xs_5: x_data_train[:, 10:12], \
                                                xs_6: x_data_train[:, 12:14], \
                                                xs_7: x_data_train[:, 14:16], \
                                                xs_8: x_data_train[:, 16:18], \
                                                xs_9: x_data_train[:, 18:20], \
                                                xs_10: x_data_train[:, 20:22], \
                                                xs_11: x_data_train[:, 22:24], \
                                                xs_12: x_data_train[:, 24:26], \
                                                xs_13: x_data_train[:, 26:28], \
                                                ys: y_data_train.reshape([len(y_data_train), 1])})
    train_loss.append(train_temp_loss)

    if ((i + 1)%200 == 0):
        # to see the step improvement
        print('Training Steps: ' + str(i + 1) + '. Training Loss = ' + str(train_temp_loss))

    # testing
    test_temp_loss = sess.run(loss, feed_dict={xs_0: x_data_test[:, 0:2], \
                                               xs_1: x_data_test[:, 2:4], \
                                               xs_2: x_data_test[:, 4:6], \
                                               xs_3: x_data_test[:, 6:8], \
                                               xs_4: x_data_test[:, 8:10], \
                                               xs_5: x_data_test[:, 10:12], \
                                               xs_6: x_data_test[:, 12:14], \
                                               xs_7: x_data_test[:, 14:16], \
                                               xs_8: x_data_test[:, 16:18], \
                                               xs_9: x_data_test[:, 18:20], \
                                               xs_10: x_data_test[:, 20:22], \
                                               xs_11: x_data_test[:, 22:24], \
                                               xs_12: x_data_test[:, 24:26], \
                                               xs_13: x_data_test[:, 26:28], \
                                               ys: y_data_test.reshape([len(y_data_test), 1])})
    test_loss.append(test_temp_loss)

#===============================================================================
# Use matplotlib to display loss 
#===============================================================================
plt.plot(train_loss, 'k-', label='Train Loss')
plt.plot(test_loss, 'r--', label='Test Loss')
plt.title('Loss (RMSE) per Training Step')
#plt.legend(loc='best',prop={'size':12})
plt.legend(loc='best')
plt.xlabel('Training Step')
plt.ylabel('Loss (in kcal/mol)')
plt.savefig('Loss.pdf')
#plt.show()

