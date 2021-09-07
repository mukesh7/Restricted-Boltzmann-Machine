import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from loader import save_mnist_image
def sigmoid(a):
    return 1/(np.exp(-a)+1)

'''
Hyperparameters
'''
batch_size = 50
epoch = 1
hid_size = 100
K = 2
lr = 0.01

'''
Preprocessing for MNIST dataset
'''
data = pd.read_csv('train.csv')
data_val = pd.read_csv('val.csv')
images_val = data_val.iloc[:,1:785].values
images_val = images_val.astype(np.float)
val_labels_flat = data_val['label'].values.ravel()
images = data.iloc[:,1:785].values
images = images.astype(np.float)
labels_flat = data['label'].values.ravel()
images = np.vstack((images,images_val))
labels = np.append(labels_flat, val_labels_flat)
(r,col) = np.shape(images)
for i in range(r):
    for j in range(col):
        if(images[i,j] >= 127):
            images[i,j] = 1
        else:
            images[i,j] = 0


'''
Code
'''
#np.random.seed(1234)
vis_size = col
W = 0.1*np.random.randn(vis_size,hid_size)
b = np.zeros((vis_size,1))
c = -4.0*np.ones((hid_size,1))
cntr = 0
for i in range(epoch):
    for j in range(int(r/batch_size)):
        for k in range(batch_size):
            v0 = np.reshape(images[k+cntr,:], (col,1))
            for t in range(K):
                htmp = sigmoid(np.add(np.matmul(W.T,v0),c))
                for u in range(hid_size):
                    r_num = np.random.rand()
                    if(r_num > htmp[u,0]):
                        htmp[u] = 0
                    else:
                        htmp[u] = 1
                vtmp = sigmoid(np.add(np.matmul(W,htmp),b))
                for u in range(vis_size):
                    r_num = np.random.rand()
                    if(r_num > vtmp[u,0]):
                        vtmp[u] = 0
                    else:
                        vtmp[u] = 1
            Wup = np.dot(lr,np.subtract(np.matmul(sigmoid(np.add(np.matmul(W.T,v0),c)), np.reshape(v0,((1,-1)))) , np.matmul(sigmoid(np.add(np.matmul(W.T,vtmp),c)), np.reshape(vtmp,((1,-1)))) ))
            bup = np.dot(lr, np.subtract(v0,vtmp))
            cup = np.dot(lr,np.subtract(sigmoid(np.add(np.matmul(W.T,v0),c)), sigmoid(np.add(np.matmul(W.T,vtmp),c)) ))
            W = np.add(W,Wup.T)
            b = np.add(b,bup)
            c = np.add(c,cup)
        cntr += batch_size
        print("batch ",j," : num ",k)
    print("DONE",j)
    
    
def save_weights(list_of_weights):
      with open('weights.pkl', 'wb') as f:
             pickle.dump(list_of_weights, f)

def load_weights():
      with open('/home/mukesh/dl_rbm/weights.pkl','rb') as f:
              list_of_weights = pickle.load(f)
      return list_of_weights
save_weights([W,b,c])




data_tst = pd.read_csv('/home/mukesh/dl_rbm/test.csv')
'''
Test data
'''
[W,b,c] = load_weights();
images_tst = data_tst.iloc[:,1:785].values
images_tst = images_tst.astype(np.float)
img_tst = data_tst.iloc[:,1:785].values
img_tst = images_tst.astype(np.float)
(r,col) = np.shape(images_tst)
for i in range(r):
    for j in range(col):
        if(images_tst[i,j] >= 127):
            images_tst[i,j] = 1
        else:
            images_tst[i,j] = 0
for i in range(5):
    v = np.reshape(images_tst[i,:], (col,1))
    htmp = sigmoid(np.add(np.matmul(W.T,v),c))
    for u in range(hid_size):
        r_num = np.random.rand()
        if(r_num > htmp[u,0]):
            htmp[u] = 0
        else:
            htmp[u] = 1
    img = np.add(np.matmul(W,htmp), b)
    save_mnist_image(np.reshape(img_tst[i,:],(28,28)), "Output_Gibbs", str(i) + "original.png")
    save_mnist_image(np.reshape(img,(28,28)), "Output_Gibbs", str(i) + "reconstructed.png")














