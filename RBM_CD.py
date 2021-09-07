import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def sigmoid(a):
    return 1/(np.exp(-a)+1)

def save_weights(list_of_weights):
      with open('lr1.pkl', 'wb') as f:
             pickle.dump(list_of_weights, f)

def load_weights():
      with open('/home/mukesh/dl_rbm/q12.pkl','rb') as f:
              list_of_weights = pickle.load(f)
      return list_of_weights
  
'''
Hyperparameters
'''
batch_size = 50
epoch = 10
hid_size = 100
#K = 1
lr = 0.01

'''
Preprocessing for MNIST dataset
'''
data = pd.read_csv('/home/mukesh/dl_rbm/train.csv')
data_val = pd.read_csv('/home/mukesh/dl_rbm/val.csv')
images_val = data_val.iloc[:,1:785].values
images_val = images_val.astype(np.float)
val_labels_flat = data_val['label'].values.ravel()
images = data.iloc[:,1:785].values
images = images.astype(np.float)
labels_flat = data['label'].values.ravel()
images = np.vstack((images,images_val))
#images = images[0:10000,:]
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
K_list = [1]
err_list = []
for K in K_list:
    err_list_tmp = []
    vis_size = col
    itr = 0
    sample = []
    W = 0.1*np.random.randn(vis_size,hid_size)
    b = np.ones((vis_size,1))
    c = np.zeros((hid_size,1))
    cntr = 0
    for i in range(epoch):
        err = 0
        cntr=0
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
                            
                itr += 1
                err_tmp = np.sum((v0-vtmp)**2)
                err += err_tmp
            cntr += batch_size
            if(j%20==0):
                print("Batch Completed = ",j,",K = ",K)
        err /= r
        err_list_tmp.append(err)
        print("Epoch Completed = ", i, ", Err = ", err)
    err_list.append(err_list_tmp)
save_weights([err_list])

#'''
#Test data
##'''
##[W,b,c] = load_weights();
#data_tst = pd.read_csv('/home/mukesh/dl_rbm/test.csv')
#images_tst = data_tst.iloc[:,1:785].values
#images_tst = images_tst.astype(np.float)
#img_tst = data_tst.iloc[:,1:785].values
#img_tst = images_tst.astype(np.float)
#(r,col) = np.shape(images_tst)
#hid_rep= []
#for i in range(r):
#    for j in range(col):
#        if(images_tst[i,j] >= 127):
#            images_tst[i,j] = 1
#        else:
#            images_tst[i,j] = 0
#for i in range(r):
#    v = np.reshape(images_tst[i,:], (col,1))
#    htmp = sigmoid(np.add(np.matmul(W.T,v),c))
#    for u in range(hid_size):
#        r_num = np.random.rand()
#        if(r_num > htmp[u,0]):
#            htmp[u] = 0
#        else:
#            htmp[u] = 1
#    hid_rep.append(htmp)
#    img = np.add(np.matmul(W,htmp), b)
#save_weights([hid_rep])
#    plt.figure()
#    plt.imshow(np.reshape(img_tst[i,:],(28,28)),cmap='gray')
#    plt.figure()
#    plt.imshow(np.reshape(img,(28,28)), cmap = 'gray')

#'''
#Plot
#'''
#fig = plt.figure()
#spec = gridspec.GridSpec(ncols=8, nrows=8)
#for i in range(1,64):
#    print(i)
#    a = np.asarray(sample[i])
#    fig.add_subplot(spec[int(i/8), int(i%8)])
#    plt.imshow(np.reshape(a,(28,28)), cmap = 'gray')



#recons_img = np.reshape(hid_rep,(10000,5))
#X = np.array(recons_img)
#xx = pd.read_csv('/home/mukesh/dl_rbm/test-sol.csv')
#xx = np.asmatrix(xx)
#labs  = np.array(xx[:,1])
#X_embedded = TSNE(n_components=2).fit_transform(X)
#X_embedded.shape
#xy = np.zeros((2, 1000))
#xy[0] = X_embedded[:,0]
#xy[1] = X_embedded[:,1]
#colors = [i for i in labs[0:10000]]
#colors = np.reshape(colors,(10000,))
#plt.scatter(X_embedded[:,0], X_embedded[:,1], c=colors)
#plt.show()

#target_ids = range(10)
#plt.figure(figsize=(6, 5))
#colors = 'grey', 'red', 'hotpink', 'yellow', 'olivedrab', 'darkgreen', 'darkgoldenrod', 'royalblue', 'darkviolet', 'cyan'
#for i, c in zip(target_ids, colors):
#    print(i)
#    plt.scatter(X_embedded[np.where(labs == i), 0], X_embedded[np.where(labs == i), 1], c=c, alpha = 1-(i/20))
#plt.legend(['T-shirt/Top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle-boot'],loc=3)
#plt.show()



















