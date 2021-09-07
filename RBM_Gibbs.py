import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
#from loader import save_mnist_image


def sigmoid(a):
    return 1/(np.exp(-a)+1)

def save_weights(list_of_weights):
      with open('v_sam.pkl', 'wb') as f:
             pickle.dump(list_of_weights, f)

def load_weights():
      with open('v_sam.pkl','rb') as f:
              list_of_weights = pickle.load(f)
      return list_of_weights
  
'''
Hyperparameters
'''
batch_size = 100
epoch = 5
hid_size = 100
K = 3
R = 1
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
#images = images[0:1000,:]
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
v = np.zeros((col,R))
v_sam=[]
for i in range(epoch):
    cntr = 0
    for j in range(int(r/batch_size)):
        for k in range(batch_size):
            v0 = np.zeros((col,1))
            vd = np.reshape(images[k+cntr,:], (col,1))
            for i in range(col):
                if(np.random.uniform() > 0.5):
                    v0[i,0] = 1
            for t in range(K+R):
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
                if(t >= K):
                    v[:,t-K] = vtmp[:,0]
                if(j==0 and k==11 and t>=K):
                    v_sam.append(vtmp)
            Wup = np.matmul(sigmoid(np.add(np.matmul(W.T,vd),c)), np.reshape(vd,((1,-1))))
            Wtmp = np.zeros((hid_size,vis_size))
            ctmp = np.zeros((hid_size,1))
            for i in range(R):
                Wtmp = np.add(Wtmp, np.matmul( sigmoid(np.add(np.matmul(W.T, np.reshape(v[:,i],(vis_size,1)) ),c)), np.reshape(v[:,i],((1,-1))) ))
                ctmp = np.add(ctmp, sigmoid(np.add(np.matmul(W.T, np.reshape(v[:,i],(vis_size,1)) ),c)))
            Wtmp /= R
            ctmp /= R
            Wup = np.dot(lr, np.subtract(Wup,Wtmp))
            
            btmp = np.zeros((vis_size,1))
            for i in range(R):
                btmp = np.add(btmp, np.reshape(v[:,i],(784,1)))
            btmp /= R
            bup = np.dot(lr, np.subtract(vd,btmp))
            
            cup = sigmoid(np.add(np.matmul(W.T,vd),c))
            cup = np.dot(lr, np.subtract(cup,ctmp))
            
            W = np.add(W,Wup.T)
            b = np.add(b,bup)
            c = np.add(c,cup)
        cntr += batch_size
        print("batch ",j," : num ",k)
    print("DONE",j)
save_weights([v_sam])

'''
Test data
'''
[W,b,c] = load_weights();
data_tst = pd.read_csv('test.csv')
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
for i in range(3):
    v = np.reshape(images_tst[41,:], (col,1))
    htmp = sigmoid(np.add(np.matmul(W.T,v),c))
    for u in range(hid_size):
        r_num = np.random.rand()
        if(r_num > htmp[u,0]):
            htmp[u] = 0
        else:
            htmp[u] = 1
    img = np.add(np.matmul(W,htmp), b)
    plt.figure()
#    save_mnist_image(np.reshape(img_tst[i,:],(28,28)), "Output_Gibbs", str(i) + "original.png")
    plt.imshow(np.reshape(img_tst[41,:],(28,28)),cmap='gray')
    plt.figure()
#    save_mnist_image(np.reshape(img,(28,28)), "Output_Gibbs", str(i) + "reconstructed.png")
    
    plt.imshow(np.reshape(img,(28,28)), cmap = 'gray')