import os
import numpy as np
import struct
'''
define the load_data function, the function is to load the data from the binary file
the labels shape is (60000,)
the images shape is (47040000,)
'''
def load_data(path,kind="train"):
    labels_path = os.path.join(path,"%s-labels.idx1-ubyte"%kind)
    images_path = os.path.join(path,"%s-images.idx3-ubyte"%kind)
    with open(labels_path,'rb') as labpath:
        magic,n = struct.unpack(">II",labpath.read(8))
        labels = np.fromfile(labpath,dtype=np.uint8)
    with open(images_path,'rb') as imgpath:
        magic,nums,rows,cols = struct.unpack(">IIII",imgpath.read(16))
        images = np.fromfile(imgpath,dtype=np.uint8)
    return labels,images
orig_labels,orig_images = np.array(load_data("data/",kind="train"))
'''
the last layer's output will be a softmax function, so change the labels' shape to (10,60000)
there are 0,1,2,3...,9 ,ten number which is handwriting, change them to vector :
                    for example: 1---------[0,1,0,0,0,0,0,0,0,0,0].T
                                 5---------[0,0,0,0,0,1,0,0,0,0,0].T
                    explain the get_one_hot function:
                            the np.eye(nb_classes) will generate a matrix like this: nb_classes=4
                                                [1,0,0,0]
                                                [0,1,0,0]
                                                [0,0,1,0]
                                                [0,0,0,1]
                            and if there are 4 handwriting number :0,1,2,3
                            np.array(targets).reshape(-1) = [0,1,1,3,2,1,0,1,2]
                            if np.eye(nb_classes)[1]-------[0,1,0,0]
                               np.eye(nb_classes)[2]-------[0,0,1,0]
                            so the function will generate the one_hot function
'''
def get_one_hot(targets,nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]
'''
the images shape is (60000,784)
the labels shape is (60000,10)
'''
images = orig_images.reshape(60000,784)/255
labels = get_one_hot(orig_labels,10)
'''
the input images shape should be (784,60000)
the output labels shape should be (10,60000)
so we should change the images and labels shape
'''
input_images = images.T
output_labels = labels.T
'''
define the hyper-parameters : the filter size, the padding size, the stride, the pooling size
'''
hyper_parameters={"filter_size":4,
                  "pooling_size":6,
                  "stride":1}
def filter_initializer(hyper_parameters):
    filter_size = hyper_parameters["filter_size"]
    w = np.random.rand(filter_size,filter_size)
    b = np.zeros(())
    filter_parameter ={"filter_w":w,
                       "filter_b":b}
    return filter_parameter
'''
the convolution function include two steps:
                (1)--------computer the value of the convolution about one slice matrix, and return a number z
                (2)--------the convolution function will return the matrix z 
'''
def single_conv_step(a_slice,filter_w,filter_b):
    s = a_slice*filter_w
    z = np.sum(s)
    z = z+float(filter_b)
    return z
def conv_forward(A_prev,filter_parameters,hyper_parameters):
    (m,n_H_prev,n_W_prev) = A_prev.shape
    filter_size = hyper_parameters["filter_size"]
    stride = hyper_parameters["stride"]
    n_H = int((n_H_prev-filter_size)/stride)+1
    n_W = int((n_W_prev-filter_size)/stride)+1
    filter_w = filter_parameters["filter_w"]
    filter_b = filter_parameters["filter_b"]
    Z = np.zeros((n_H,n_W))
    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_H):
            for w in range(n_W):
                vert_start = h*stride
                vert_end = vert_start+filter_size
                horiz_start = w*stride
                horiz_end = horiz_start+filter_size
                a_slice_prev = a_prev[vert_start:vert_end,horiz_start:horiz_end]
                Z[i,h,w] = single_conv_step(a_slice_prev,filter_w,filter_b)
    assert(Z.shape==(n_H,n_W))
    return Z
def pooling_forward(A_prev,hyper_parameters,mode="average"):
    (m,n_H_prev,n_W_prev) = A_prev.shape
    pooling_size = hyper_parameters["pooling_size"]
    stride = hyper_parameters["stride"]
    n_H = int((n_H_prev-pooling_size)/stride)+1
    n_W = int((n_W_prev-pooling_size)/stride)+1
    A = np.zeros((m,n_H,n_W))
    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                vert_start = h*stride
                vert_end = vert_start+pooling_size
                horiz_start = w*stride
                horiz_end = horiz_start+pooling_size
                a_prev_slice = A_prev[i,vert_start:vert_end,horiz_start:horiz_end]
                if mode=="max":
                    A[i,h,w] = np.max(a_prev_slice)
                elif mode=="average":
                    A[i,h,w] = np.mean(a_prev_slice)
    assert(A.shape == (m,n_H,n_W))
    return A
'''
define the function about the pooling layer's backward
            the distribute_value:
                        [2,2,3]   avg pooling
                        [1,1,1]   ------------[2]  this is forward 
                        [4,3,1]
                so the backward is 
                        dz/(3*3)=dz/9
                        [2-dz/9,2-dz/9,3-dz/9]
                        [1-dz/9,1-dz/9,1-dz/9]
                        [4-dz/9,3-dz/9,1-dz/9]
            the distribute_value is just one step value the backward            
'''
def distribute_value(dz,shape):
    (n_h,n_w) = shape
    average = dz/(n_h*n_w)
    a = np.ones(shape)*average
    return a
def pooling_backward(dZ,A_prev,hyper_parameters):
    filter_size = hyper_parameters["filter_size"]
    (m,n_h,n_w) = dZ.shape
    dA_prev = np.zeros(A_prev.shape)
    for i in range(m):
        for h in range(n_h):
            for w in range(n_w):
                vert_start = h
                vert_end = vert_start+filter_size
                horiz_start = w
                horiz_end = horiz_start+filter_size
                dz = dZ[i,h,w]
                shape = (filter_size,filter_size)
                dA_prev[i,vert_start:vert_end,horiz_start:horiz_end]+=distribute_value(dz,shape)
    assert (dA_prev.shape == A_prev.shape)
    return dA_prev

'''
define the cnn backward function
'''
def conv_backward(dZ,A_prev,filter_w,filter_b,hyper_parameters):
    filter_size = hyper_parameters["filter_size"]
    (m,n_h,n_w) = dZ.shape
    dA_prev = np.zeros(A_prev.shape)
    dw = np.zeros((filter_w.shape))
    db = np.zeros((filter_b.shape))
    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_h):
            for w in range(n_w):
                vert_start = h
                vert_end = vert_start+filter_size
                horiz_start = w
                horiz_end = horiz_start+filter_size
                a_slice = a_prev[vert_start:vert_end,horiz_start:horiz_end]
                dA_prev[i,vert_start:vert_end,horiz_start:horiz_end]+= filter_w[:,:]*dZ[i,h,w]
                dw = a_slice*dZ[i,h,w]
                db +=dZ[i,h,w]
    assert(dA_prev.shape == A_prev.shape)
    assert(dw.shape == filter_w.shape)
    assert(db.shape == filter_b.shape)
    return dA_prev,dw,db

'''
the architecture of the neural network is :
                sigmoid            avg              reshape         input the linear nn
    input_x*filter---(60000,25,25)------(60000,20,20)----(400,60000) -----------
(60000,28,28)*(4,4)                     

define this cnn_forward

'''
def cnn_forward(input_x,hyper_parameters):
    filter_parameters = filter_initializer(hyper_parameters)
    Z1 = conv_forward(input_x,filter_parameters,hyper_parameters)
    A1 = sigmoid(Z1)
    A2 = pooling_forward(A1,hyper_parameters,mode="average")
    assert(A1.shape == (60000,25,25))
    assert(A2.shape == (60000,20,20))
    return Z1,A1,A2
def cnn_backward(dA2,input_x,A1,filter_parameters,learning_rate,hyper_parameters):
    filter_w = filter_parameters["filter_w"]
    filter_b = filter_parameters["filter_b"]
    dA1 = pooling_backward(dA2,A1,hyper_parameters)
    dZ1 = dA1*A1*(1-A1)
    _,dfilter_w,dfilter_b = conv_backward(dZ1,input_x,filter_w,filter_b,hyper_parameters)
    filter_w = filter_w-learning_rate*dfilter_w
    filter_b = filter_b-learning_rate*dfilter_b
    filter_parameters = {"filter_w":filter_w,
                         "filter_b":filter_b}
    return filter_parameters
'''
define the linear neural network architecture:
            the linear_input_layer-------(400,60000)
            the linear_hidden_layer------(30units)
            the linear_output_layer------(10,60000)
'''
def initializer_with_hidden_layers(num_hidden_units):
    w1 = np.random.randn(num_hidden_units,784)
    b1 = np.zeros((num_hidden_units,1))
    w2 = np.random.randn(10,num_hidden_units)
    b2 = np.zeros((10,1))
    parameters={"w1":w1,
                "b1":b1,
                "w2":w2,
                "b2":b2}
    return parameters
'''
define the two activative function:
                sigmoid function-----hidden layer
                softmax function-----output layer
'''
def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s
def softmax(z):
    total = np.sum(np.exp(z),axis=0,keepdims=True)
    s = np.exp(z)/total
    return s
def forward_propagation(linear_input_x,output_y,parameters):
    m = linear_input_x.shape[1]
    w1 = parameters["w1"]
    b1 = parameters["b1"]
    w2 = parameters["w2"]
    b2 = parameters["b2"]
    a1 = sigmoid(np.dot(w1,linear_input_x)+b1)
    a2 = softmax(np.dot(w2,a1)+b2)
    value_cost = -1/m*np.sum(output_y*np.log(a2))
    return a1,a2,value_cost
def backward_propagation(linear_input_x,output_y,parameters,learning_rate):
    m = linear_input_x.shape[1]
    w1 = parameters["w1"]
    b1 = parameters["b1"]
    w2 = parameters["w2"]
    b2 = parameters["b2"]
    a1,a2,cost = forward_propagation(linear_input_x,output_y,parameters)
    dz2 = a2-output_y
    dw2 = 1/m*np.dot(dz2,a1.T)
    db2 = 1/m*np.sum(dz2,axis=1,keepdims=True)
    dz1 = 1/m*np.dot(w2.T,dz2)*a1*(1-a1)
    dw1 = 1/m*np.dot(dz1,linear_input_x.T)
    db1 = 1/m*np.sum(dz1,axis=1,keepdims=True)
    dlinear_input_x = 1/m*np.dot(w1.T,dz1)
    w1 = w1-learning_rate*dw1
    b1 = b1-learning_rate*db1
    w2 = w2-learning_rate*dw2
    b2 = b2-learning_rate*db2
    assert (w1.shape==dw1.shape)
    assert (b1.shape==db1.shape)
    assert (w2.shape==dw2.shape)
    assert (b2.shape==db2.shape)
    assert (dlinear_input_x.shape == linear_input_x.shape)
    parameters={"w1":w1,
                "b1":b1,
                "w2":w2,
                "b2":b2}
    return parameters,dlinear_input_x,cost
def model(input_x,output_y,learning_rate=0.05,hyper_parameters=hyper_parameters,iterations=1000):
    filter_parameters = filter_initializer(hyper_parameters)
    parameters = initializer_with_hidden_layers(num_hidden_units=30)
    for i in range(iterations):
        Z1,A1,A2 = cnn_forward(input_x,hyper_parameters)
        A3_mid = A2.reshape(A2.shape[0],A2.shape[1]*A2.shape[2])
        A3 = A3_mid.T
        assert (A3.shape == (400,60000))
        parameters,dlinear_input_x,cost = backward_propagation(A3,output_y,parameters,learning_rate)
        assert (dlinear_input_x.shape == A3.shape)
        dA2 = (dlinear_input_x.reshape.T).reshape(60000,20,20)
        filter_parameters = cnn_backward(dA2,input_x,A1,filter_parameters,learning_rate,hyper_parameters)
        if i%100 ==0 :
            print("cost after %i iteration "%i+" is : "+str(cost))
    return filter_parameters,parameters