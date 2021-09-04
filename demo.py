import tensorflow as tf
import pandas as pd
from keras import regularizers
from keras import layers
import numpy as np
import urllib.request
import os.path
import random

#NN parameters
learning_rate = 0.005
dropout_rate = 0.3
training_epochs = 10
batch_size = 10

#download data file if not present
data_file = 'wdbc.data'
if not os.path.isfile(data_file):
    urllib.request.urlretrieve("http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/" + data_file, data_file)
train_data = pd.read_csv(data_file,  delimiter=',', index_col=0, header=None)

#add onehot labels
one_hot = pd.get_dummies(train_data.iloc[:,0])
train_data = train_data.join(one_hot)

#split data by category. B=Benign, M=Malignant
Malignant = train_data[train_data.M == 1]
Benign = train_data[train_data.B == 1]
print("Benign samples:", len(Benign), "Malignant samples:", len(Malignant))

#train set will contain 80% of malignant and 80% of benign samples
train_X = Malignant.sample(frac=0.8)
train_X = pd.concat([train_X, Benign.sample(frac=0.8, random_state = 2)], axis=0)

#text_X dataset will contain all the diagnostics not present in train_X
test_X = train_data.loc[~train_data.index.isin(train_X.index)]

#extract onehot labels from features
train_Y = train_X.iloc[:,-2:]
test_Y = test_X.iloc[:,-2:]

#drop labels from features
train_X = train_X.iloc[:,1:-2]
test_X = test_X.iloc[:,1:-2]

#feature headers
features = train_X.columns.values

#normalize features
for feature in features:
    mean, std = train_data[feature].mean(), train_data[feature].std()
    train_X.loc[:, feature] = (train_X[feature] - mean) / std
    test_X.loc[:, feature] = (test_X[feature] - mean) / std

#number of input, output
input_nodes = train_X.shape[1]
num_labels = test_Y.shape[1]

#convert table into numpy matrix
train_x = train_X.to_numpy().astype(float)
train_y = train_Y.to_numpy().astype(float)

test_x = test_X.to_numpy().astype(float)
test_y = test_Y.to_numpy().astype(float)

#split test set into test and validation sets
split = int(len(test_x) / 2)

val_x = test_x[:split]
val_y = test_y[:split]
test_x = test_x[split:]
test_y = test_y[split:]

#custom dense layer with regularization, relu activation and batch normalization
class dense_block(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(dense_block, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.dense = tf.keras.layers.Dense(
            input_shape[-1],
            activation = 'relu',
            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
            bias_regularizer=regularizers.l2(1e-4),
        )
        self.batch_norm = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        return self.batch_norm(self.dense(inputs))

def build_model(use_dropout):
    inputs = tf.keras.Input(shape=(input_nodes,), name='x')
    x = dense_block(32)(inputs)
    x = dense_block(16)(x)
    x = dense_block(16)(x)
    if use_dropout:
         x = tf.keras.layers.Dropout(dropout_rate)(x)
    y = tf.keras.layers.Dense(num_labels, activation = 'softmax', name='y')(x)
    model = tf.keras.Model(inputs=inputs, outputs=y)
    model.compile(  loss='categorical_crossentropy',
                    metrics='categorical_accuracy', 
                    optimizer='adam')
    return model

#train model with no dropout layer, and save tensorboard logs into ./tensorboard_no_dropout folder
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./tensorboard_no_dropout", histogram_freq=1)
model = build_model(use_dropout = False)
model.fit(train_x, train_y, epochs=training_epochs, batch_size=batch_size, validation_data=(val_x, val_y), callbacks=[tensorboard_callback])
_, no_dropout_acc = model.evaluate(test_x, test_y)

#train model with dropout layer, and save tensorboard logs into ./tensorboard_dropout folder
#a model with a dropout layer should achieve better accuracy on test and validation sets,
#as it reduces overfitting, and help better generalization
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./tensorboard_dropout", histogram_freq=1)
model2 = build_model(use_dropout = True)
model2.fit(train_x, train_y, epochs=training_epochs, batch_size=batch_size, validation_data=(val_x, val_y), callbacks=[tensorboard_callback])
_, droupout_acc = model2.evaluate(test_x, test_y)

#print summary. the results suffer from high extent of randomness,
#but usually the model with dropout layer achieves better accuracy
print("summary:")
print(f"model without dropout layer achieved {no_dropout_acc*100}% test accuracy.")
print(f"model with dropout layer achieved {droupout_acc*100}% test accuracy.")  

if droupout_acc == no_dropout_acc:
    print("there is no difference between the models trained with and without dropout layer")

if droupout_acc > no_dropout_acc:
    print("the model trained with dropout layer is more accurate, than the model trained without one")

if droupout_acc < no_dropout_acc:
    print("the model trained with no dropout layer is more accurate, than the model trained with one")