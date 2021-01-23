import pandas as pd
from enum import Enum
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, SimpleRNN, Flatten
from keras.activations import relu, elu, softmax
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.losses import logcosh, binary_crossentropy, categorical_crossentropy
from keras.models import load_model
from talos import Scan

import matplotlib.pyplot as plt
from keras.callbacks import Callback
import itertools
from sklearn.metrics import confusion_matrix

from dataload import fall_detection_dataset, one_y_data     ## load my data

input_dim = 5


class Actions(Enum):
    # framewise_recognition.h5
    # squat = 0
    # stand = 1
    # walk = 2
    # wave = 3

    # # framewise_recognition_under_scene.h5
    # stand = 0
    # walk = 1
    # operate = 2
    # fall_down = 3
    # # run = 4

    # framewise_recognition_check.h5
    moving = 0
    fall_down = 1


# Callback class to visialize training progress
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


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

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
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

def keras_model(x_train, y_train, x_val, y_val, params):
    # build keras model
    model = Sequential()
    model.add(SimpleRNN(units=128, input_shape=X_train.shape[1:] ,return_sequences=True))
    model.add(Dense(units=params['first_hidden_layer'], activation=params['activation1']))
    model.add(BatchNormalization())
    model.add(Dropout(params['dropout']))
    model.add(Dense(units=params['second_hidden_layer'], activation=params['activation2']))
    model.add(BatchNormalization())
    model.add(Dropout(params['dropout']))
    model.add(Dense(units=params['third_hidden_layer'], activation=params['activation3']))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(units=2, activation=params['last_activation']))  # units = nums of classes
    # model_name = "framewise_recognition_check_v4"
    # checkpoint_callback = ModelCheckpoint(model_name+'.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=20)        # 조기종료 콜백함수 정의 (과적합 발생시 조기종료)
    model.compile(loss=params['losses'], optimizer=params['optimizer'], metrics=['accuracy'])
    out = model.fit(X_train, Y_train, batch_size=params['batch_size'], epochs=params['epochs'], verbose=1, validation_data=(X_test, Y_test), callbacks=[his, early_stopping])   #early_stopping, checkpoint_callback
                                                                                                                 
    return out, model                                                                                                       

parm = {'activation1':['relu', 'elu'],
     'activation2':['relu', 'elu'],
     'activation3':['relu', 'elu'],
     'last_activation': ['softmax'],
     'optimizer': ['Adam', "RMSprop"],
     'losses': ['logcosh', 'binary_crossentropy', 'categorical_crossentropy'],
     'first_hidden_layer': [128, 64, 32],
     'second_hidden_layer': [64, 32, 16],
     'third_hidden_layer': [32, 16, 8],
     'dropout': [.2, .3, .4],
     'batch_size': [32, 64, 128, 256],
     'epochs': [32, 128, 256]}


# load data
raw_data = pd.read_csv('check.csv', header=0)
dataset = raw_data.values
# X = dataset[:, 0:36].astype(float)
# Y = dataset[:, 36]
# X = dataset[0:3289, 0:36].astype(float)  # 忽略run数据
# Y = dataset[0:3289, 36]
############################ chage ############################
# X = dataset[0:3289, 0:30].astype(float)  # 忽略run数据
# Y = dataset[0:3289, 30]
dataset_m = fall_detection_dataset(22,8)
X = dataset_m[:,:,:28]
print(X.shape)
Y = one_y_data(dataset_m)
encoder_Y = []
for y in Y:
    encoder_Y.append(y)
############################ chage ############################
# 将类别编码为数字
# encoder = LabelEncoder()
# encoder_Y = encoder.fit_transform(Y)
# print(encoder_Y[0], encoder_Y[900], encoder_Y[1800], encoder_Y[2700])
# encoder_Y = [0]*744 + [1]*722 + [2]*815 + [3]*1008 + [4]*811
# encoder_Y = [0]*744 + [1]*722 + [2]*815 + [3]*1008                        # <==
# one hot 编码
dummy_Y = np_utils.to_categorical(encoder_Y)

# train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_Y, test_size=0.1, random_state=9)

from keras.callbacks import EarlyStopping, ModelCheckpoint
his = LossHistory()
# call keras model
# training
h = Scan(x=X_train, 
         y=Y_train,
         x_val=X_test,
         y_val=Y_test,
         params=parm,
         experiment_name='first_test',
         model=keras_model,
         fraction_limit=0.1)


# from talos.utils.recover_best_model import recover_best_model
# results, models = recover_best_model(x=X_train, 
#                                      y=Y_train,
#                                      x_val=X_test,
#                                      y_val=Y_test,
#                                      experiment_log='minimal_iris.csv',
#                                      input_model=keras_model,
#                                      n_models=5,
#                                      task='multi_label')

# analyze_object = talos.Analyze(scan_object)
# analyze_object.data
# analyze_object.round()
# # get the highest result for any metric
# analyze_object.high('val_acc')

# # get the round with the best result
# analyze_object.rounds2high('val_acc')

# # get the best paramaters
# analyze_object.best_params('val_acc', ['acc', 'loss', 'val_loss'])

# # get correlation for hyperparameters against a metric
# analyze_object.correlate('val_loss', ['acc', 'loss', 'val_loss'])
# # a regression plot for two dimensions 
# analyze_object.plot_regs('val_acc', 'val_loss')

# # line plot
# analyze_object.plot_line('val_acc')

# # up to two dimensional kernel density estimator
# analyze_object.plot_kde('val_acc')

# # a simple histogram
# analyze_object.plot_hist('val_acc', bins=50)

# # heatmap correlation
# analyze_object.plot_corr('val_loss', ['acc', 'loss', 'val_loss'])

# # a four dimensional bar grid
# analyze_object.plot_bars('batch_size', 'val_acc', 'first_neuron', 'lr')


early_stopping = EarlyStopping(monitor='val_loss', patience=20)        # 조기종료 콜백함수 정의 (과적합 발생시 조기종료)

model.summary()
his.loss_plot('epoch')
# model.save('framewise_recognition_check_v6.h5')

# # evaluate and draw confusion matrix
# print('Test:')
# score, accuracy = model.evaluate(X_test,Y_test,batch_size=32)
# print('Test Score:{:.3}'.format(score))
# print('Test accuracy:{:.3}'.format(accuracy))
# # confusion matrix
# Y_pred = model.predict(X_test)
# cfm = confusion_matrix(np.argmax(Y_test,axis=1), np.argmax(Y_pred, axis=1))
# np.set_printoptions(precision=2)
#
# plt.figure()
# class_names = ['squat', 'stand', 'walk', 'wave']
# plot_confusion_matrix(cfm, classes=class_names, title='Confusion Matrix')
# plt.show()

# # test
# model = load_model('framewise_recognition.h5')
#
# test_input = [0.43, 0.46, 0.43, 0.52, 0.4, 0.52, 0.39, 0.61, 0.4,
#               0.67, 0.46, 0.52, 0.46, 0.61, 0.46, 0.67, 0.42, 0.67,
#               0.42, 0.81, 0.43, 0.91, 0.45, 0.67, 0.45, 0.81, 0.45,
#               0.91, 0.42, 0.44, 0.43, 0.44, 0.42, 0.46, 0.44, 0.46]
# test_np = np.array(test_input)
# test_np = test_np.reshape(-1, 36)
#
# test_np = np.array(X[1033]).reshape(-1, 36)
# if test_np.size > 0:
#     pred = np.argmax(model.predict(test_np))
#     init_label = Actions(pred).name
#     print(init_label)
