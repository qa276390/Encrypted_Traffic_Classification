
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import tensorflow as tf
config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
import argparse
import keras
from keras import models
from keras import layers
from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence
from keras.models import Sequential, Model, load_model
from keras.layers import Concatenate, Dense, LSTM, Input, concatenate, Dropout
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import normalize 
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
from keras import backend as K
from sklearn.cross_validation import StratifiedKFold
import sklearn
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from tensorflow.python.client import device_lib
import sys
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
import itertools

# In[2]:

def next_batch(num, data, labels):
    num_el = data.shape[0]
    while True: # or whatever condition you may have
        idx = np.arange(0 , num_el)
        np.random.shuffle(idx)
        current_idx = 0
        while current_idx < num_el:
            batch_idx = idx[current_idx:current_idx+num]
            current_idx += num
            data_shuffle = [data[ i,:] for i in batch_idx]
            labels_shuffle = [labels[ i] for i in batch_idx]
            yield np.asarray(data_shuffle), np.asarray(labels_shuffle)

            
def neural_net(x, dropout):
  
    # Store layers weight & bias
    initializer = tf.contrib.layers.xavier_initializer()
    weights = {
        'h1': tf.Variable(initializer([num_input, n_hidden_1])),
        'h2': tf.Variable(initializer([n_hidden_1, n_hidden_2])),
        'h3': tf.Variable(initializer([n_hidden_2, n_hidden_3])),
        'h4': tf.Variable(initializer([n_hidden_3, n_hidden_4])),
        'h5': tf.Variable(initializer([n_hidden_4, n_hidden_5])),
        'h6': tf.Variable(initializer([n_hidden_5, n_hidden_6])),
        'h7': tf.Variable(initializer([n_hidden_6, n_hidden_7])),
        'out': tf.Variable(initializer([n_hidden_7, num_classes]))
    }
    biases = {
        'b1': tf.Variable(initializer([n_hidden_1])),
        'b2': tf.Variable( initializer([n_hidden_2])),
        'b3': tf.Variable( initializer([n_hidden_3])),
        'b4': tf.Variable( initializer([n_hidden_4])),
        'b5': tf.Variable( initializer([n_hidden_5])),
        'b6': tf.Variable( initializer([n_hidden_6])),
        'b7': tf.Variable( initializer([n_hidden_7])),
        'out': tf.Variable( initializer([num_classes]))
    }
    
    
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.layers.dropout(tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1'])), dropout[0])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.layers.dropout(tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])), dropout[1])
    # Hidden fully connected layer with 256 neurons
    layer_3 = tf.layers.dropout(tf.nn.relu(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])), dropout[2])
    # Hidden fully connected layer with 256 neurons
    layer_4 = tf.layers.dropout(tf.nn.relu(tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])), dropout[3])
    # Hidden fully connected layer with 256 neurons
    layer_5 = tf.layers.dropout(tf.nn.relu(tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])), dropout[4])
    layer_6 = tf.layers.dropout(tf.nn.relu(tf.add(tf.matmul(layer_5, weights['h6']), biases['b6'])), dropout[4])
    layer_7 = tf.layers.dropout(tf.nn.relu(tf.add(tf.matmul(layer_6, weights['h7']), biases['b7'])), dropout[4])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_7, weights['out']) + biases['out']
    return out_layer

def bn_neural_net(x, dropout):
  
    # Store layers weight & bias
    initializer = tf.contrib.layers.xavier_initializer()
    weights = {
        'h1': tf.Variable(initializer([num_input, n_hidden_1])),
        'h2': tf.Variable(initializer([n_hidden_1, n_hidden_2])),
        'h3': tf.Variable(initializer([n_hidden_2, n_hidden_3])),
        'h4': tf.Variable(initializer([n_hidden_3, n_hidden_4])),
        'h5': tf.Variable(initializer([n_hidden_4, n_hidden_5])),
        'h6': tf.Variable(initializer([n_hidden_5, n_hidden_6])),
        'h7': tf.Variable(initializer([n_hidden_6, n_hidden_7])),
        'out': tf.Variable(initializer([n_hidden_7, num_classes]))
    }
    biases = {
        'b1': tf.Variable(initializer([n_hidden_1])),
        'b2': tf.Variable( initializer([n_hidden_2])),
        'b3': tf.Variable( initializer([n_hidden_3])),
        'b4': tf.Variable( initializer([n_hidden_4])),
        'b5': tf.Variable( initializer([n_hidden_5])),
        'b6': tf.Variable( initializer([n_hidden_6])),
        'b7': tf.Variable( initializer([n_hidden_7])),
        'out': tf.Variable( initializer([num_classes]))
    }
    
    
    # Hidden fully connected layer with 256 neurons
    x = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = tf.layers.dropout(x, dropout[0])
    # Hidden fully connected layer with 256 neurons
    x = tf.add(tf.matmul(x, weights['h2']), biases['b2'])
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = tf.layers.dropout(x, dropout[1])
    # Hidden fully connected layer with 256 neurons
    x = tf.add(tf.matmul(x, weights['h3']), biases['b3'])
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = tf.layers.dropout(x, dropout[2])
    # Hidden fully connected layer with 256 neurons
    x = tf.add(tf.matmul(x, weights['h4']), biases['b4'])
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = tf.layers.dropout(x, dropout[3])
    # Hidden fully connected layer with 256 neurons
    x = tf.add(tf.matmul(x, weights['h5']), biases['b5'])
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = tf.layers.dropout(x, dropout[4])
    # Hidden fully connected layer with 256 neurons
    x = tf.add(tf.matmul(x, weights['h6']), biases['b6'])
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = tf.layers.dropout(x, dropout[4])
    # Hidden fully connected layer with 256 neurons
    x = tf.add(tf.matmul(x, weights['h7']), biases['b7'])
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = tf.layers.dropout(x, dropout[4])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(x, weights['out']) + biases['out']
    return out_layer


# In[12]:


reDirect = False

n_hidden_1 = 1280 # 1st layer number of neurons
n_hidden_2 = 960 # 2nd layer number of neurons
n_hidden_3 = 640 # 3rd layer number of neurons
n_hidden_4 = 640 # 4th layer number of neurons
n_hidden_5 = 480 # 5th layer number of neurons
n_hidden_6 = 320 # 5th layer number of neurons
n_hidden_7 = 320 # 5th layer number of neurons
num_input = 886 # MNIST data input (img shape: 28*28)
num_classes = 12 # MNIST total classes (0-9 digits)


LABEL2DIG = {'chat':0, 'voip':1, 'trap2p':2, 'stream':3, 'file_trans':4, 'email':5, 'vpn_chat':6, 'vpn_voip':7, 'vpn_trap2p':8, 'vpn_stream':9, 'vpn_file_trans':10, 'vpn_email':11}
DIG2LABEL = {v: k for k, v in LABEL2DIG.items()}
nclass = 12

#random.seed(0)
np.random.seed(0)
tf.set_random_seed(0)


# In[3]:




def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
print(get_available_gpus())


# In[5]:



if(reDirect):
    old_stdout = sys.stdout
    sys.stdout = open( FOLDER + '/log', 'w')
print(os.getpid())


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.figure(figsize = (10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def DNN(opts):
    
    FOLDER = opts.output_path
    if not os.path.exists(FOLDER):
        os.mkdir(FOLDER)
    MODEL_PATH = FOLDER + '/model.ckpt'
    FIG_PATH = FOLDER + '/Confusion_Matrix.png'
    FIG_PATH_N = FOLDER + '/Confusion_Matrix_Norm.png'

    X_train = np.load(opts.source_data_folder+'/X_train.npy')
    y_train = np.load(opts.source_data_folder+'/y_train.npy')
    X_train = X_train.astype('float32') 
    #y_train = y_train
    print('X_train:', np.shape(X_train))
    print('y_train:', np.shape(y_train))
    clw = []
    nums = np.zeros(6)
    maxsize = 0
    print('-'*20)
    for cat in np.unique(y_train):
        size = np.shape(np.where(y_train==cat))[1]
        #nums[LABEL2DIG[cat]] = size
        print(DIG2LABEL[cat]+": "+str(size))
        clw.append(1/size)
        if(size > maxsize):
            maxsize = size
    print('-'*20)

    clw = [i*maxsize for i in clw]

    y = y_train
    y_train = to_categorical(y_train, num_classes = nclass)

    X_train = normalize(X_train, norm='max', axis=0, copy=True, return_norm=False)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

    dim = np.shape(X_train)[1]
    print(dim)
    size = np.shape(X_train)[0]


    # Parameters
    lr = 1e-3
    num_steps = 1000000
    batch_size = opts.batch_size 
    n_batches = int(size/batch_size)
    display_step = int(size/batch_size)
    patience = opts.patience


    # Network Parameters
    global num_input
    global num_classes
    num_input = dim # MNIST data input (img shape: 28*28)
    num_classes = nclass # MNIST total classes (0-9 digits)

    # Create model



    global_step = tf.Variable(0, trainable=False)
    with tf.device('/gpu:0'):
        # tf Graph input
        X = tf.placeholder("float", [None, num_input])
        Y = tf.placeholder("float", [None, num_classes])
        dropout = tf.placeholder(tf.float32, [5, 1])
        logits = bn_neural_net(X, dropout)
        prediction = tf.nn.softmax(logits)
         
        class_weights = tf.constant([clw])
        weights = tf.reduce_sum(class_weights * Y, axis = 1)
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y)
        weighted_losses = unweighted_losses * weights
        # Define loss and optimizer
        loss_op = tf.reduce_mean(weighted_losses)
        #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        #train_op = optimizer.minimize(loss_op)


        # Evaluate model
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        lr = tf.train.exponential_decay(1e-2, global_step=global_step,decay_steps = 20000, decay_rate=0.5, staircase = True)
        add_global = global_step.assign_add(1)
        
        optimizer = tf.train.GradientDescentOptimizer(lr)
        #optimizer = tf.train.AdamOptimizer(lr)
        with tf.control_dependencies([add_global]):
            train_op = optimizer.minimize(loss_op)
    # Start training
    saver = tf.train.Saver()


    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()


    # In[ ]:


    best_acc = 0
    train_acc = 0
    pat = 0
    list_train = []
    list_test = []
    #drop = [0.7, 0.75, 0.8, 0.9, 0.9]
    drop = [0.5, 0.5, 0.5, 0.3, 0.2]
    drop = np.reshape(drop, [5, 1])
    drop_0 = [0.0, 0.0, 0.0, 0.0, 0.0]
    drop_0 = np.reshape(drop_0, [5, 1])
    n_batch_size = batch_size
    end = 0
    with tf.Session(config=config) as sess:
        
        tf.global_variables_initializer().run()
        print(sess.run(lr))
        next_batch_gen = next_batch(batch_size, X_train, y_train)
        for i in range(1, num_steps+1):
            if end == 1:
                break
            for j in range(n_batches):
            #n_batch_size = int(batch_size*num_steps/1000)
                batch_x, batch_y = next(next_batch_gen)
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, dropout: drop})
                train_acc += sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y, dropout: drop_0})
                if(j == n_batches-1):
                    # Calculate batch loss and accuracy
                    train_acc /= (n_batches)
                    clr, loss, acc = sess.run([lr, loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y, dropout: drop_0})
                    val_acc = sess.run(accuracy, feed_dict={X: X_test, Y: y_test, dropout: drop_0})
                    print("Step " + str(i) + ", Minibatch Loss= " +                       "{:.4f}".format(loss) + ", Training Accuracy= " +                       "{:.4f}".format(train_acc) + ", Testing Accuracy= " +                       "{:.4f}".format(val_acc) + ", Learning Rate=" +                       "{:.4f}".format(clr))
                    #print(n_batch_size)
                    list_train.append(train_acc)
                    list_test.append(val_acc)
                    train_acc = 0
                    #print("Testing Accuracy:", val_acc)
                    if(val_acc > best_acc):
                        save_path = saver.save(sess, MODEL_PATH)
                        print("Model saved in path: %s" % save_path)
                        best_acc = val_acc
                        pat = 0
                    else:
                        pat+=1
                    if(pat>=patience):
                        print("Early Stop!")
                        end = 1
                        break

        print("Optimization Finished!")

        # Calculate accuracy for MNIST test images
        print("Testing Accuracy:",         sess.run(accuracy, feed_dict={X: X_test, Y: y_test, dropout: drop_0}))

    sess=tf.Session()    
    y_pred = []
    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, save_path)
        print("Model restored.")
        y_pred = sess.run(prediction, feed_dict={X: X_test, Y: y_test, dropout: drop_0})
        val_acc = sess.run(accuracy, feed_dict={X: X_test, Y: y_test, dropout: drop_0})
        print("Testing Accuracy= " + "{:.4f}".format(val_acc))



    y_p = y_pred.argmax(axis = -1)
    y_t = y_test.argmax(axis = -1)
    class_names = [DIG2LABEL[i] for i in range(nclass)]
    print(class_names)
    cnf_matrix = confusion_matrix(y_t, y_p)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,title='Confusion matrix, without normalization')
    plt.savefig(FIG_PATH)

    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,title='Normalized confusion matrix')
    plt.savefig(FIG_PATH_N)


    print('f1-scroe = {}'.format(f1_score(y_t, y_p, average=None)))
    print('prcision = {}'.format(precision_score(y_t, y_p, average=None)))
    print('recall = {}'.format(recall_score(y_t, y_p, average=None)))  
    print('macro f1 = {}'.format(f1_score(y_t, y_p, average='macro')))


    # In[ ]:
    
def XGB(opts):
    reDirect = False
    FOLDER = 'clean_vpn12_xgb'
    if not os.path.exists(FOLDER):
        os.mkdir(FOLDER)
    MODEL_PATH = FOLDER + '/model.h5'
    FIG_PATH = FOLDER + '/Confusion_Matrix.png'
    FIG_PATH_N = FOLDER + '/Confusion_Matrix_Norm.png'


    import sys
    if(reDirect):
        old_stdout = sys.stdout
        sys.stdout = open( FOLDER + '/log', 'w')

    X_train = np.load(opts.source_data_folder+'/X_train.npy')
    y_train = np.load(opts.source_data_folder+'/y_train.npy')
    X_train = X_train.astype('float32') 

    print('X_train:', np.shape(X_train))
    print('y_train:', np.shape(y_train))

    maxsize = 0
    print('-'*20)
    for cat in np.unique(y_train):
        size = np.shape(np.where(y_train==cat))[1]
        print(str(cat)+": "+str(np.shape(np.where(y_train==cat))[1]))
        if(size > maxsize):
            maxsize = size
    print('-'*20)

    y = y_train

    X_train = normalize(X_train, norm='l2', axis=0, copy=True, return_norm=False)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)


    dim = np.shape(X_train)[1]
    print(dim)
    
    #Setting Classifier
    xgbc = XGBClassifier(max_depth=20, tree_method='exact',  n_estimators=180, n_jobs=-1)
    #training
    xgbc.fit(X_train, y_train,eval_set=[(X_train, y_train), (X_val, y_val)], early_stopping_rounds=30, verbose = True)


    results = xgbc.score(X_test, y_test)

    print('Test accuracy: ', results)

    if(reDirect):
        sys.stdout = old_stdout
    print('Test accuracy: ', results)

    xgbc.get_booster().save_model(MODEL_PATH)

    y_pred = xgbc.predict(X_test)

    #load the best model
    import xbgoost as xgb
    bst = xgb.Booster({'nthread': 4})  # init model
    bst.load_model(MODEL_PATH)  # load data
    y_pred = bst.predict(X_test)



    y_p = y_pred
    y_t = y_test
    class_names = [DIG2LABEL[i] for i in range(nclass)]
    cnf_matrix = confusion_matrix(y_t, y_p)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,title='Confusion matrix, without normalization')
    plt.savefig(FIG_PATH)

    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,title='Normalized confusion matrix')
    plt.savefig(FIG_PATH_N)

    print('f1-scroe = {}'.format(f1_score(y_t, y_p, average=None)))
    print('prcision = {}'.format(precision_score(y_t, y_p, average=None)))
    print('recall = {}'.format(recall_score(y_t, y_p, average=None)))  
    print('macro f1 = {}'.format(f1_score(y_t, y_p, average='macro')))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--mode', type=str,
                        default='DNN', dest='mode',
                        help='DNN or XGB')
    
    parser.add_argument('--source_data_folder', type=str,
                        default='../data', dest='source_data_folder',
                        help='Path to source data')
    parser.add_argument('--output_folder', type=str,
                        default='./log/log.txt', dest='output_path',
                        help='Path to output')
    
    parser.add_argument('--batch_size', type=int,
                        default='1024', dest='batch_size',
                        help='batch_size')
    parser.add_argument('--patience', type=int,
                        default='1000', dest='patience',
                        help='patience')
    

    opts = parser.parse_args()

    #y = S2P(opts.source_data_path)
    if(opts.mode=='XGB'):
        print('XGB...')
        XGB(opts)
    else:
        print('DNN...')
        DNN(opts)
