#!D:/workplace/python

# -*- coding: utf-8 -*-

# @File  : handwriting_STFTCNN.py

# @Author: Li Qingpei

# @Date  : 2018/11/26

# @Software: PyCharm
import skimage.io as io
from skimage import data_dir
import os
import scipy.io as sio
from sklearn.cross_validation import train_test_split
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone
from sklearn import tree
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
# from xgboost import XGBClassifier
from sklearn import feature_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
# import skflow
import tensorflow as tf
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,LSTM,Flatten
from keras.layers import Conv1D,GlobalAveragePooling1D,MaxPooling1D,AveragePooling1D,GlobalMaxPooling1D,MaxPooling2D,Conv2D,GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model
from keras.callbacks import TensorBoard
import keras
import xlwt
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectPercentile,f_classif,RFE
from keras import backend as K
from sklearn.externals import joblib
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image

matfn=u'E:/声音小组/LQPHP/audiotrack/matlab_code/handwriting/20181206_lqp_812_ultraGesture_pauseRecordDemo/label.mat'
data = sio.loadmat(matfn)

# 批量读取灰度图片
data_dir='E:/声音小组/LQPHP/audiotrack/matlab_code/handwriting/20181206_lqp_812_ultraGesture_pauseRecordDemo/gray_picture_1024_512_1024'
str1=data_dir + '/*.jpg'
coll = io.ImageCollection(str1)
print(len(coll))

# 生成标签
# 生成标签
# x = [1] * 260
# for n in range(0, 260):
#     x[n] = math.ceil(float(n / 10))
#     print('x[n]\n', x[n])
# y=[1]*260
# for n in range(0, 259):
#     y[n] = x[n + 1]
# y[259] = 26

label = data['label']
# label = np.transpose(label)
img=[]
colll=[]
for n in range(0,len(coll)):
    print('n:\n',n)
    im = Image.fromarray(coll[n])
    w,h = im.size
    im1 = im.thumbnail((w//2,h//2))
    img1 = np.array(im)
    img = img1.reshape(-1,img1.shape[0],img1.shape[1],1)
    # img = coll[n].reshape(-1,coll[n].shape[0],coll[n].shape[1],1).astype('float32')
    colll.append(img)
# img = coll.reshape(-1,1,Width,Height)
collll = np.concatenate(colll,axis=0)
# label = np.array(y)
# colll=np.array(coll)
print('collll.shape:',collll.shape)
print('label.shape',label.shape)

#del img

#del coll
#del colll

train_X, test_X, train_y, test_y = train_test_split(collll,
                                                    label,
                                                    test_size=0.3,
                                                    random_state=0)
#del collll
print('label:',label)
print('label.shape:',label.shape)
#======使用两个数据集，一个是训练集，一个是测试集


print('data has readed!!!')
N = 26  # N为类别个数
acc=np.zeros((8,1))
re=[]
ma=[]


#***************** preprocessing.scale*******************************************************************************************
# X_train_scaled = preprocessing.scale(train_X)
# X_test_scaled = preprocessing.scale(test_X)

#  一个样本一个样本的变换
# X_train_scaled = np.zeros((train_X.shape[0], train_X.shape[1]))
# X_test_scaled = np.zeros((test_X.shape[0], test_X.shape[1]))
#
# for i in range(0, train_X.shape[0]):
#     X_train_scaled[i, :] = preprocessing.scale(train_X[i, :])
#
# for i in range(0, train_X.shape[0]):
#     X_test_scaled[i, :] = preprocessing.scale(test_X[i, :])



#***************** MinMaxScaler*******************************************************************************************
# train_X = np.transpose(train_X)
# test_X = np.transpose(test_X)
# mms = MinMaxScaler().fit(train_X)
# X_train_norm = mms.fit_transform(train_X)
# X_test_norm = mms.transform(test_X)
#
# X_train_norm = np.transpose(X_train_norm)
# X_test_norm = np.transpose(X_test_norm)
#



#一个样本一个样本的变换
#for i in range(0,train_X.shape[0]):
#X_train_scaled[i,:] =  mms.fit_transform(train_X[i,:])
#
#
#for i in range(0,test_X.shape[0]):
#X_test_scaled[i,:] = mms.fit_transform(test_X[i,:])
# X_train_std=train_X
# X_test_std=test_X

#*****************StandardScaler*******************************************************************************************
# stdsc = StandardScaler().fit(train_X)
# X_train_std = stdsc.transform(train_X)
# X_test_std = stdsc.transform(test_X)
# X_train_std = np.transpose(X_train_std)
# X_test_std = np.transpose(X_test_std)
#
# print('X_train_std shape:',X_train_std.shape)
# #plt.plot(train_X[1,:],color='blue')
# plt.plot(X_train_std[1,:],color='red')
# plt.show()

#*****************Non-linear transformation*******************************************************************************************


#
# train_X = np.transpose(train_X)
# test_X = np.transpose(test_X)

#*****************SelectKBest*******************************************************************************************
# CH2=SelectKBest(chi2,k=1000)
# X_train_SK = CH2.fit_transform(train_X,train_y)
# X_test_SK = CH2.transform(test_X)


#*****************L1-based feature selection*******************************************************************************************
# lsvc = LinearSVC(C=0.01,penalty='l1',dual=False).fit(train_X,train_y)
# model_l1 = SelectFromModel(lsvc,prefit=True)
# X_train_L1 = model_l1.transform(train_X)
# X_test_L1 = model_l1.transform(test_X)


#*****************L2-based feature selection*******************************************************************************************
# lsvc_l2 = LinearSVC(C=0.01,penalty='l2',dual=False).fit(train_X,train_y)
# model_l2 = SelectFromModel(lsvc,prefit=True)
# X_train_L2 = model_l2.transform(train_X)
# X_test_L2 = model_l2.transform(test_X)

#*****************Tree-based feature selection*******************************************************************************************
# clfTree = ExtraTreesClassifier()
# clfTree = clfTree.fit(train_X,train_y)
# model_clf_Tree = SelectFromModel(clfTree,prefit=True)
# X_train_tree = model_clf_Tree.transform(train_X)
# X_test_tree = model_clf_Tree.transform(test_X)



#*****************Removing features with low variance*******************************************************************************************
# sel = VarianceThreshold(threshold=3).fit(train_X,train_y)
# X_train_thre = sel.fit_transform(train_X)
# X_test_thre = sel.transform(test_X)


#*****************Univariate Feature Selection*******************************************************************************************
# selector = SelectPercentile(percentile=50)
# X_train_P = selector.fit_transform(train_X,train_y)
# X_test_P = selector.transform(test_X)


#*****************特征递归消除法&&&&&&&&&&&(未测试，特别慢)&&&&&&&&&&&&&&*******************************************************************************************
# LRFE = RFE(estimator=LogisticRegression(), n_features_to_select=500).fit_transform(train_X, train_y)
# X_train_LRFE = RFE.fit_transform(train_X,train_y)
# X_test_LRFE = RFE.transform(test_X)


# X_train_std = X_train_std
#
# X_test_std = X_test_std
#print('X_train_std:',X_train_std.shape)
#print('X_test_std:',X_test_std.shape)



# 将一维label转换为二维label
train_Y = keras.utils.to_categorical((train_y - 1), num_classes=N)
test_Y = keras.utils.to_categorical((test_y - 1), num_classes=N)


##===================9.CNN===============================================================================================

# data_X = np.reshape(train_X, (train_X.shape[0],train_X.shape[1],train_X.shape[2], 1)).astype('float32')
# data_Y = np.reshape(test_X, (test_X.shape[0],test_X.shape[1],test_X.shape[2], 1)).astype('float32')
# img = img.reshape(-1,1,Width,Height)
data_X = train_X/255
data_Y = test_X/255

#imgGen = ImageDataGenerator(featurewise_center=True,featurewise_std_normalization=True)
# imgGen = ImageDataGenerator(zca_whitening=True) #ZCA白化
# imgGen = ImageDataGenerator(shear_range=0.2)   #图像剪切
# imgGen.fit(data_X)
# imgGen.fit(data_Y)

model = Sequential()

print('data_X.shape',data_X.shape)


#======================LeNet==================================================

#
# model.add(Conv2D(32,(5,5),strides=(1,1),input_shape=(data_X.shape[1], data_X.shape[2],1),padding='valid',activation='relu',kernel_initializer='uniform'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Conv2D(64,(5,5),strides=(1,1),padding='valid',activation='relu',kernel_initializer='uniform'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Flatten())
# model.add(Dense(100,activation='relu'))
# model.add(Dense(N,activation='softmax'))
# optimizer_Adamax=keras.optimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)
# model.compile(optimizer=optimizer_Adamax,loss='categorical_crossentropy',metrics=['accuracy'])
# model.summary()
#======================结束LeNet==================================================

#======================ZfeNet==================================================
#
model = Sequential()
model.add(Conv2D(32,(6,6),strides=(2,2),input_shape=(data_X.shape[1], data_X.shape[2],1),padding='same',activation='tanh',kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(4,4),strides=(2,2)))
# model.add(Dropout(0.2))
model.add(Conv2D(32,(6,6),strides=(1,1),padding='same',activation='tanh',kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(64,(6,6),strides=(2,2),padding='same',activation='tanh',kernel_initializer='uniform'))
# model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
# model.add(Conv2D(32,(6,6),strides=(1,1),padding='same',activation='tanh',kernel_initializer='uniform'))#没有这一层是89%
model.add(Conv2D(64,(6,6),strides=(1,1),padding='same',activation='tanh',kernel_initializer='uniform'))
model.add(Conv2D(32,(6,6),strides=(1,1),padding='same',activation='tanh',kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256,activation='tanh'))
model.add(Dropout(0.75))
model.add(Dense( 128,activation='tanh'))
model.add(Dropout(0.75))
# model.add(LSTM(units=100))
model.add(Dense(N,activation='softmax'))
optimizer_Adamax=keras.optimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)
model.compile(loss='categorical_crossentropy',optimizer=optimizer_Adamax,metrics=['accuracy'])  ##optimizer='Adamax'60%,,          'Nadam','adam','SGD','RMSprop','Adadelta','TFOptimizer',      'Adagrad',
model.summary()
#======================结束ZfeNet==================================================

#
#
# #======================VGG-13==================================================
# model.add(Conv2D(32,(3,3),strides=(1,1),input_shape=(data_X.shape[1], data_X.shape[2],1),padding='same',activation='tanh',kernel_initializer='uniform'))
# model.add(Conv2D(32,(3,3),strides=(1,1),padding='same',activation='tanh',kernel_initializer='uniform'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='tanh',kernel_initializer='uniform'))
# model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='tanh',kernel_initializer='uniform'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='tanh',kernel_initializer='uniform'))
# model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='tanh',kernel_initializer='uniform'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='tanh',kernel_initializer='uniform'))
# model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='tanh',kernel_initializer='uniform'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# # model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='tanh',kernel_initializer='uniform'))
# # model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='tanh',kernel_initializer='uniform'))
# # model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Flatten())
# model.add(Dense(256,activation='tanh'))
# model.add(Dropout(0.5))
# model.add(Dense(128,activation='tanh'))
# model.add(Dropout(0.5))
# model.add(Dense(N,activation='softmax'))
# model.compile(loss='categorical_crossentropy',optimizer='adamax',metrics=['accuracy'])
# model.summary()
#======================结束VGG-13==================================================

#======================AlexNet==================================================
#
# model.add(Conv2D(32,(3,3),strides=(2,2),input_shape=(data_X.shape[1], data_X.shape[2],1),padding='valid',activation='tanh',kernel_initializer='uniform'))
# model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
# model.add(Conv2D(32,(3,3),strides=(1,1),padding='same',activation='tanh',kernel_initializer='uniform'))
# model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
# model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='tanh',kernel_initializer='uniform'))
# model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='tanh',kernel_initializer='uniform'))
# model.add(Conv2D(32,(3,3),strides=(1,1),padding='same',activation='tanh',kernel_initializer='uniform'))
# model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
# model.add(Flatten())
# model.add(Dense(512,activation='tanh'))
# model.add(Dropout(0.5))
# model.add(Dense(256,activation='tanh'))
# model.add(Dropout(0.5))
# model.add(Dense(N,activation='softmax'))
# model.compile(loss='categorical_crossentropy',optimizer='Adamax',metrics=['accuracy'])
# model.summary()

#
# # print('data_X.shape[2]',data_X.shape[2])
# # print('data_X.shape[0]',data_X.shape[0])
# model.add(Conv2D(filters=96, kernel_size=(3,3), activation='relu', input_shape=(data_X.shape[1], data_X.shape[2],1)))
# # model.add(Dropout(0.002))
# model.add(Conv2D(filters=256, kernel_size=(5,5), activation='relu'))
# model.add(Conv2D(filters=384, kernel_size=(3,3), activation='relu'))
# model.add(Conv2D(filters=384, kernel_size=(3,3), activation='relu'))
# model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu'))
# # model.add(Dropout(0.002))
# model.add(MaxPooling2D(2,2))
# model.add(Dropout(0.5))
# model.add(BatchNormalization())
# # model.add(Conv1D(64, 3, activation='tanh'))
# # model.add(Dropout(0.02))
# # model.add(Conv1D(64, 3, activation='tanh'))
# # model.add(Flatten())
# # model.add(Dropout(0.2))
# model.add(MaxPooling2D(pool_size=(2,2)))
# # model.add(GlobalMaxPooling2D())
# # model.add(Dropout(0.5))
# # model.add(BatchNormalization())
# # model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
# model.add(Flatten())
# # model.add(Dropout(0.2))
# # model.add(Dense(64,activation='tanh'))
# # model.add(Dense(64,activation='tanh'))
# model.add(Dense(32,activation='tanh'))
# # model.add(Dense(64,activation='tanh'))
# # model.add(Dense(64,activation='tanh'))
#
# model.add(Dropout(0.4))
# # model.add(BatchNormalization())
# model.add(Dense(N, activation='softmax'))
#
#
# #plot_model(model,to_file='model.png')
# #损失函数：loss='mean_squared_error','mean_absolute_error','mean_absolute_percentage_error','sparse_categorical_crossentropy','binary_crossentropy'
# #   loss = 'mean_squared_logarithmic_error','squared_hinge','hinge','categorical_hinge','logcosh','categorical_crossentropy','kullback_leibler_divergence'
# # loss='poisson','cosine_proximity'
# #optimizer='adam','SGD','RMSprop','Adagrad','Adadelta','Adamax','Nadam','TFOptimizer'
# #model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# #keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.3, amsgrad=False)
# # keras.optimizers.SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)
#
# model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
#

tb=TensorBoard(log_dir='./logs',histogram_freq=1,batch_size=6,write_graph=True,write_grads=False,write_images=False,embeddings_freq=0,embeddings_layer_names=None,embeddings_metadata=None)
callbacks=[tb]


history=model.fit(data_X,train_Y,epochs=1000,batch_size=10,validation_split=0.2)
scoress=model.evaluate(data_Y,test_Y)
scoress_train = model.evaluate(data_X,train_Y)
Y=model.predict(data_Y)
print('Y[0]',Y.shape[0])
print('Y[1]',Y.shape[1])
print('scoress',scoress)
print('scoress_train',scoress_train)


# summarize history for accuracy
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('CNN-model-accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.tight_layout()
plt.savefig('./accuracyVSepoch.png')
plt.show()
# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('CNN-model-loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.tight_layout()
plt.savefig('./lossVSepoch.png')
plt.show()



np.set_printoptions(threshold=1000000000)
print('Score:', scoress[1])
print('Score_train:', scoress_train)
acc[7]=scoress[1]
X = model.predict(data_Y)
#print('Predict:', X)
#print('true label:', test_y - 1)
# 将得到的概率转换为真实标签,也就是得到每一行的最大值的下标
XX = np.zeros((X.shape[0], 1))
for i in range(X.shape[0]):
    XX[i, 0] = np.argmax(X[i, :])
#print('predict_labels:', XX+1)
Se_prob=model.predict_proba(data_X)  # 输出分类概率
re.append(classification_report(test_y,(XX+1)))
ma.append(confusion_matrix(test_y, (XX+1)))
print('report length:',len(re))
print('matrix length:',len(ma))
joblib.dump(model, "train_model.m")
model = joblib.load("train_model.m")  #加载模型
#提取网络中第14层输出作为特征   CNN
get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[10].output])
X_test_std = get_3rd_layer_output([data_Y])[0]
X_train_std = get_3rd_layer_output([data_X])[0]
print('layer_output_test.shape',X_test_std.shape)
sio.savemat('wxy_saveddata.mat', {'CNN_data_test': X,'CNN_data_train': X_data_X,'X_test_std':X_test_std,'X_train_std': X_train_std})


#============1.LogisticRegression=========================================================================================
lr = LogisticRegression(multi_class='multinomial', penalty='l2', solver='sag', C=100)
lr.fit(X_train_std, train_y)
print('LogisticRegression')
print('Training accuracy:', lr.score(X_train_std, train_y))
print('Test accyracy:', lr.score(X_test_std, test_y))
acc[0]=lr.score(X_test_std, test_y)
lr_prob=lr.predict_proba(X_test_std)  # 输出分类概率
re.append(classification_report(test_y, lr.predict(X_test_std)))
ma.append(confusion_matrix(test_y, lr.predict(X_test_std)))
cr = LogisticRegression(multi_class='multinomial', penalty='l2', solver='sag', C=100)
# cr.fit(X_train_reduced, train_y)
# accuracy = cross_val_score(cr, X_test_std, test_y.ravel(), cv=10)
# accTrain = cross_val_score(cr, X_train_std, train_y.ravel(), cv=10)
# print('Test accyracy:{}\n{}', np.mean(accuracy), accuracy)
# print("Train accuracy:{}\n{}", np.mean(accTrain), accTrain)
lrPredictlabel=lr.predict(X_test_std)

#=============2.SVM=======================================================================================================
# kernel='linear' ,'rbf' kernel='precomputed'
classifier = svm.SVC(C=1, kernel='linear', gamma=0.1, decision_function_shape='ovr', probability=True)
# classifier = svm.SVR()
classifier.fit(X_train_std, train_y)
pred_label = classifier.predict(X_test_std)
print('svm')
print('Training accuracy:', classifier.score(X_train_std, train_y))
print('Test accyracy:', classifier.score(X_test_std, test_y))
acc[1]=classifier.score(X_test_std, test_y)
svmPredictlabel = classifier.predict(X_test_std)
svm_prob=classifier.predict_proba(X_test_std)  # 输出分类概率
y_true,y_pred = test_y,classifier.predict(X_test_std)
re.append(classification_report(test_y, classifier.predict(X_test_std)))
ma.append(confusion_matrix(test_y, classifier.predict(X_test_std)))
print(classification_report(y_true,y_pred))
print(confusion_matrix(y_true,y_pred))


#==========3.KNN==========================================================================================================
#algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}
knn = KNeighborsClassifier(n_neighbors=12,algorithm='brute',weights='distance',)
#sbs = SBS(knn,k_features=1)
#sbs.fit(X_train_std,train_y)
knn.fit(X_train_std,train_y)
print('knn')
print('Training accuracy:',knn.score(X_train_std,train_y))
print('Test accyracy:',knn.score(X_test_std,test_y))
acc[2]=knn.score(X_test_std,test_y)
re.append(classification_report(test_y, knn.predict(X_test_std)))
ma.append(confusion_matrix(test_y, knn.predict(X_test_std)))
knnPredictLabel=knn.predict(X_test_std)
knn_prob=knn.predict_proba(X_test_std)  # 输出分类概率


#========4.DecisionTreeClassifier=========================================================================================
#=========http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
#deClf = tree.DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=100,max_features='auto')
deClf = tree.DecisionTreeClassifier(criterion='entropy')
deClf.fit(X_train_std,train_y)
print('DecisionTree')
print('Training accuracy:',deClf.score(X_train_std,train_y))
print('Test accuracy:',deClf.score(X_test_std,test_y))
acc[3]=deClf.score(X_test_std,test_y)
re.append(classification_report(test_y, deClf.predict(X_test_std)))
ma.append(confusion_matrix(test_y, deClf.predict(X_test_std)))
dePredictLabel = deClf.predict(X_test_std)
deClf_prob=deClf.predict_proba(X_test_std)  # 输出分类概率

#========5.GaussianNB=====================================================================================================
modelbys = GaussianNB()
modelbys.fit(X_train_std,train_y)
print('GaussianNB')
print('Training accuracy:',modelbys.score(X_train_std,train_y))
print('Test accuracy:',modelbys.score(X_test_std,test_y))
acc[4]=modelbys.score(X_test_std,test_y)
re.append(classification_report(test_y, modelbys.predict(X_test_std)))
ma.append(confusion_matrix(test_y, modelbys.predict(X_test_std)))
NBPredicted = modelbys.predict(X_test_std)
NB_prob=modelbys.predict_proba(X_test_std)  # 输出分类概率

#==========6.RandomForestClassifier=======================================================================================
modeRFC = RandomForestClassifier(max_depth=20, random_state=0)
modeRFC.fit(X_train_std,train_y)
print('RandomForestClassifier:')
print('Training accuracy:',modeRFC.score(X_train_std,train_y))
print('Test accuracy:',modeRFC.score(X_test_std,test_y))
acc[5]=modeRFC.score(X_test_std,test_y)
re.append(classification_report(test_y, modeRFC.predict(X_test_std)))
ma.append(confusion_matrix(test_y, modeRFC.predict(X_test_std)))
RFCPredictLabel = modeRFC.predict(X_test_std)
RFC_prob=modeRFC.predict_proba(X_test_std)  # 输出分类概率

#===========7.GradientBoostingClassifier================================================================================
modelGBC = GradientBoostingClassifier(n_estimators = 10,learning_rate=0.01,max_depth=1,random_state=0)
modelGBC.fit(X_train_std,train_y)
print('GradientBoostingClassifier')
print('Training accuracy:',modelGBC.score(X_train_std,train_y))
print('Test accuracy:',modelGBC.score(X_test_std,test_y))
acc[6]=modelGBC.score(X_test_std,test_y)
re.append(classification_report(test_y, modelGBC.predict(X_test_std)))
ma.append(confusion_matrix(test_y, modelGBC.predict(X_test_std)))
GBCPredictLabel=modelGBC.predict(X_test_std)
GBC_prob=modelGBC.predict_proba(X_test_std)  # 输出分类概率





#==========将预测标签写入excel中=========================================================================================
data=xlwt.Workbook()
table = data.add_sheet('7_number',cell_overwrite_ok=True)

NN=4
for i in range(NN,test_y.shape[0]+NN):
    #print('IIIIII:',i)
    table.write(i,0,float(lrPredictlabel[i-NN]))
    table.write(i,1,float(svmPredictlabel[i-NN]))
    table.write(i, 2, float(knnPredictLabel[i-NN]))
    table.write(i, 3, float(dePredictLabel[i-NN]))
    table.write(i, 4, float(NBPredicted[i-NN]))
    table.write(i, 5, float(RFCPredictLabel[i-NN]))
    table.write(i, 6, float(GBCPredictLabel[i-NN]))
    table.write(i, 7, float(XX[i-NN]+1))
    table.write(i, 8, float(test_y[i-NN]))
class_name=['LogisticRegression','SVM','KNN','DecisionTree','GaussionNB','RandomForest','GradientBoosting','Sequential']

for j in range(0,8):
    table.write(0,j,class_name[j])
    table.write(1,j,float(acc[j]*100))
    table.write(2,j,re[j])
    table.write(3,j,str(ma[j]))

table.write(0,8,'true_label')


data.save('E:/声音小组\LQPHP/audiotrack/matlab_code/mousedetect/data/806HPhxy_9种识别/wxy_812_ultraGesture_handwriting_CNN10feature_512_test03_RMSprop.xls')


