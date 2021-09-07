# Restricted-Boltzmann-Machine
A restricted Boltzmann machine (RBM) is a generative stochastic artificial neural network that can learn a probability distribution over its set of inputs.

# Gibbs Sampling
In case of training RBM with Block Gibbs Sampling we start from random image and run the chain for large number of times untill we start seeing samples from: P(V,H).
In practice, Gibbs Sampling can be very inefficient because for every step of stochastic gradient descent we need to run the Markov chain for many many steps and then compute the expectation using the samples drawn from this chain.

# RBM Training using Contrastive Divergence
In case of training RBM using Contrastive Divergence instead of starting the Markov Chain at a random point we start from v^{(t)} where v^{(t)} is the current training instance.

# t-SNE Plot of MNIST Data

## 100 Dimensional Representation in 2D
![q1_100](https://user-images.githubusercontent.com/17472092/132380022-fb018345-a79b-453e-b055-e7fa38b6ee8f.png)
## 250 Dimensional Representation in 2D
![q1_250](https://user-images.githubusercontent.com/17472092/132379989-abf11755-08c2-46f4-9be9-69ecd1c93493.png)
