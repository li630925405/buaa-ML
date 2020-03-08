# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 19:49:16 2019

@author: 27221
"""

import numpy as np
import tensorflow as tf
import  cv2
from PIL import Image
#TRAIN_FILE = 'training.csv'
TEST_FILE = 'test.csv'
SAVE_PATH = './model2' #模型保存路径
VALIDATION_SIZE = 100    #验证集大小 TRAIN_NUM * 0.3
EPOCHS = 1           #迭代次数
BATCH_SIZE = 64          #每个batch大小，稍微大一点的batch会更稳定
EARLY_STOP_PATIENCE = 20 #控制early stopping的参数
TRAIN_NUM=1000
TEST_NUM=100

def input_data(test=False):
    print("input_Data...")
    X_size=[]
    Y_size=[]
    X=[]
    num=TEST_NUM if test else TRAIN_NUM
    file_dir="test/" if test else "train/"
    for i in range(num):
        
        file="{0:0>4d}".format(i)+".jpg"
        path=file_dir+file
        I=Image.open(path)
        L=I.convert("L")
        x=np.array(L)
        y_size,x_size=x.shape
        X_size.append(x_size)
        Y_size.append(y_size)
        x=cv2.resize(x,(96,96))
        
       
        x=x/255.0
        x=x.reshape((96,96,1))
        X.append(x)
    X=np.array(X)
    
    
    if test:
        y=None
    else:
        y=[]
        for i in range(num):
            x_size=X_size[i]
            y_size=Y_size[i]
            ys=[]
            y_file="train/{0:0>4d}".format(i)+".pts"
            fp=open(y_file,"r+")
            for line in fp.readlines():
                pos_x,pos_y=line.split(",")
                pos_x=float(pos_x)/x_size
                pos_y=float(pos_y)/y_size
                ys.append(pos_y)
                ys.append(pos_x)
            y.append(ys)
        y=np.array(y)
    return X,y,X_size,Y_size

    
    '''   
    file_name = TEST_FILE if test else TRAIN_FILE
    df = pd.read_csv(file_name)
    cols = df.columns[:-1]
    #dropna()是丢弃有缺失数据的样本，这样最后7000多个样本只剩2140个可用的。
    df = df.dropna()    
    df['Image'] = df['Image'].apply(lambda img: np.fromstring(img, sep=' ') / 255.0)
    X = np.vstack(df['Image'])
    X = X.reshape((-1,96,96,1))
    if test:
        y = None
    else:
        y = df[cols].values / 96.0       #将y值缩放到[0,1]区间
    return X, y '''


#根据给定的shape定义并初始化卷积核的权值变量
def weight_variable(shape,namew='w'):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial,name=namew)

#根据shape初始化bias变量
def bias_variable(shape,nameb='b'):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial,name=nameb)

#定义卷积操作
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='VALID')
#定义池化操作
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')


#保存模型函数
def save_model(saver,sess,save_path):
    path = saver.save(sess, save_path)
    print('model save in :{0}'.format(path))
    

if __name__ == "__main__":
    with tf.Session(graph=tf.Graph()) as sess:
        new_saver = tf.train.import_meta_graph('./model/my-model-0.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('./model/'))
        #定义的模型，生成pb文件需要对输入x和输出y命名，这里将输入x命名为input，输入y命名为output，同时如果有dropout参数也需进行命名。
        x = tf.placeholder("float", shape=[None, 96, 96,1],name='input')#输入占位
        y_ = tf.placeholder("float", shape=[None, 196],name='y')#输出占位
        keep_prob = tf.placeholder("float",name='keep_prob')#dropout概率值
        W_conv1 = weight_variable([3, 3, 1, 32],'w1')#32个3*3*1的卷积核
        b_conv1 = bias_variable([32],'b1')
        h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)
        W_conv2 = weight_variable([2, 2, 32, 64],'w2')#64个3*3*32的卷积核
        b_conv2 = bias_variable([64],'b2')
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)
        W_conv3 = weight_variable([2, 2, 64, 128],'w3')#128个3*3*32的卷积核
        b_conv3 = bias_variable([128],'b3')
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
        h_pool3 = max_pool_2x2(h_conv3)
        W_fc1 = weight_variable([11 * 11 * 128, 500],'wf1')#全连接层
        b_fc1 = bias_variable([500],'bf1')
        h_pool3_flat = tf.reshape(h_pool3, [-1, 11 * 11 * 128])#把第三层卷积的输出一维向量
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
        W_fc2 = weight_variable([500, 500],'wf2')
        b_fc2 = bias_variable([500],'bf2')
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2,name='hfc2')
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
        W_fc3 = weight_variable([500, 196],'wf3')
        b_fc3 = bias_variable([196],'bf3')
        y_conv = tf.add(tf.matmul(h_fc2_drop, W_fc3) + b_fc3,0.0,name='output')
        #以均方根误差为代价函数，Adam为
        rmse = tf.sqrt(tf.reduce_mean(tf.square(y_ - y_conv)))
        train_step = tf.train.AdamOptimizer(1e-3).minimize(rmse)

        #变量都要初始化
        sess.run(tf.initialize_all_variables())
        X,y,X_size,Y_size = input_data()
        X_valid, y_valid = X[:VALIDATION_SIZE], y[:VALIDATION_SIZE]
        X_train, y_train = X[VALIDATION_SIZE:], y[VALIDATION_SIZE:]

        best_validation_loss = 1000000.0
        current_epoch = 0
        TRAIN_SIZE = X_train.shape[0]
        train_index = list(range(TRAIN_SIZE))
        np.random.shuffle(train_index)
        X_train, y_train = X_train[train_index], y_train[train_index]

        saver = tf.train.Saver()

        print('begin training..., train dataset size:{0}'.format(TRAIN_SIZE))
        for i in range(EPOCHS):
            #进行每一轮训练都需将模型的'input','keep_prob','output'保存。
            #constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['input','keep_prob','output'])

            np.random.shuffle(train_index)  #每个epoch都shuffle一下效果更好
            X_train, y_train = X_train[train_index], y_train[train_index]

            for j in range(0,TRAIN_SIZE,BATCH_SIZE):
                print ('epoch {0}, train {1} samples done...'.format(i,j))

                train_step.run(feed_dict={x:X_train[j:j+BATCH_SIZE],
                    y_:y_train[j:j+BATCH_SIZE], keep_prob:0.5})


            train_loss = rmse.eval(feed_dict={x:X_train, y_:y_train, keep_prob: 1.0})
            validation_loss = rmse.eval(feed_dict={x:X_valid, y_:y_valid, keep_prob: 1.0})
            print('epoch {0} done! validation loss:{1}'.format(i, train_loss*96.0))
            print('epoch {0} done! validation loss:{1}'.format(i, validation_loss*96.0))
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                current_epoch = i
                #保存pb文件。
                #with tf.gfile.FastGFile(SAVE_PATH+'model.pb', mode='wb') as f: #模型的名字是model.pb
                 #   f.write(constant_graph.SerializeToString())

            elif (i - current_epoch) >= EARLY_STOP_PATIENCE:
                print ('early stopping')
                break

            saver.save(sess,"model/my-model",global_step=i)

        X_test,y_test,X_size,Y_size=input_data(test=True)
        for i in range(len(X_test)):
            x_test=X_test[i]
            x_size=X_size[i]
            y_size=Y_size[i]
            #img=cv2.imread()
            #print(type(x_test))
            #print(x_test)

            predictions=sess.run(y_conv,feed_dict={x:np.reshape(x_test,[-1,96,96,1]), keep_prob: 1.0})
            pt = np.vstack(np.split(predictions[0],98)).T
            x_test=cv2.resize(x_test,(x_size,y_size))
            #x_test=x_test.reshape([96,96])
            x_test=x_test*255
            for i in range(98):
                #print(pt[0][i]*96/x_size,pt[1][i]*96/y_size)
                cv2.circle(x_test,(int(pt[1][i]*x_size),int(pt[0][i]*y_size)),2, (255, 0, 0),-1)
                #cv2.circle(x_test,(pt[1][i]*96,pt[0][i]*96),2, (255, 0, 0),-1)
            x_img=Image.fromarray(x_test)
            x_img.show()
