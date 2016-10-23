# Training Neural Networks, Part I
## Finetuning
**Q. Do ConvNets need a lot of data to train?**
**A. No! it's myth.**

We rarely ever train ConvNets from scratch. Amount of variation depends on size of prepared data set.
- **small dataset** : Treat CNN as fixed feature extractor, then retain only the classifier (ex: train only last sofxmax layer)
- **medium sized dataset** : Finetuning. Train part of or entire network with CNN parameter as initial parameters   

## History
### Frank Rosenblatt (~1957, perceptron)
In perceptron, activation function is **binary step function** depends on sign of weighted sum. So it is non-differentiable function. Therefore back propagation is impossible.
### Widrow, Hoff (~1960, Adaline/Madaline)
They first use multi-layer perceptron and first data-driven approach. It's huge conceptual leap!!
### Rumelhart et al (~1986)
They first suggest concept of **back-propagation**. But performance with deeper/larger network is bad compared with other machine learning algorithm.
### Hinton, Salakhutdinov (2006)
They suggest **unsupervised pre-training** called **Boltzman machine**. In pre-training process, parameters are initialized targeting to unsupervised objective. Then forward / backward propagation work well in even deeper/larger network.
### Microsoft (2010)
First strong results appeared, winning any other machine learning algorithm in VSR(Vocabulary Speech Recognition).   
### Krizhevsky, Hinton (2012)
They overwhelming other machine learning algorithms even more dramatically!

## Overview
### One time setup
- activation function
- preprocessing
- weight initialization
- regulairzation
- gradient checking

### Training dynamics
- babysitting the learning process
- parameter updates
- hyperparameter optimization

### Evaluation
- model ensembles

## Activation function
### Sigmoid
**pros** Sigmoid activation was historically popular. And it is nice interpretation as a saturating **firing rate** of a neuron.
**cons** Saturated neurons kill the gradients. Sigmoid outputs are not zero-centered. Consider all-positive or all-negative inputs for sigmoid activation, then gradients should also be **same sign**. It causes **zig-zag** movement during parameter updates in paramter space and slower convergence. Moreover, exp() operation is a bit compute expensive.
### tanh (Lecun et al, 1991)
**pros** It achieves zero-centered.
**cons** Still, it kill gradients when saturated.
### ReLU (Krizhevsky et al, 2012)
= **Re**ctified **L**inear **U**nit
```python
# it's very simple
ReLU(x) = max(0, x)
```
**pros** It does not saturate, very computationally efficient, converges much faster than sigmoid/tanh in practice (6x).
**cons** Again it's output is not zero-centered. And ReLU neuron easily died, and will never updated. This problem can occur frequently by bad weight initialization or large learning rate.  
### Leaky ReLU (Mass et al 2013, He et al 2015)
```python
LReLU(x) = max(0.01*x, x)
```
**pros** It still does not saturate, computationally efficient, converges much faster than sigmoid/tanh in practice(6x). **Also it will not die**.
### Parametric ReLU
```python
# alpha is a parameter too!
PReLU(x) = max(alpha*x, x)
```
It generalizes fixed coefficient in Leaky ReLU to parameter, alpha. Alpha is also updated during back-propagation.
### Exponential ReLU (Clevert et al 2015)
```python
ELU(x) = x if x > 0 else alpha * (exp(x) - 1)
```
**pros** It takes all benefits of ReLU, does not die, and closer to zero-mean outputs
**cons** Computation requires exp() operation.
### Maxout (Goodfellow et al 2013)
```python
Maxout(x) = max(w1.dot(x) + b1, w2.dot(x) + b2) 
```
**pros** It generalizes PReLU, linear regime, does not saturate, and does not die.
**cons** It doubles the number of parameters.
### Summary
- Use **ReLU** (be careful with your learning rate)
- Try out **Leaky ReLU** / **Maxout** / **ELU**
- Try out **tanh** but don't expect much
- Don't use **sigmoid** (tanh is always better)

## Preprocessing
There are some **common** preprocessing techniques in machine learning.
- **zero-centered**
- **normalized data**
- **decorrelated data** : make covariance matrix diagonal
- **whitened data** : make covariance matrix identical

But we zero-center the data only in image domain. Subtract the **mean image**(AlexNet) or subtract **per-channel mean**(VGGNet)
## Weight Initialization
It's very important. Early computer vision research didn't work well because it didn't care much attention to weight initialization.
### W = 0 (or some constant)
All parameters in same layer will update identically. VERY BAD IDEA.
### small random numbers
It works fine for small network. But if network become deeper, it causes problem. Let's consider random number initialization with **zero mean, std=0.01** Gaussian. And evaluate network with unit gaussian input data. Then we can find out that distribution of activation in each hidden layer **narrow down to zero** as forwarding network. Zero activation in each layer lead to zero gradient, which called **vanishing gradient**.
What if we increase std=0.01 to 1.00 for weight initialization? Then weight is too big and will saturate to both extremes. It also lead to vanishing gradient.
### Xavier initialization (Glorot et al, 2010)
```python
W = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in)
```
This method initialize weights has bigger fan-in with smaller value. It's very reasonable because bigger fan-in leads to bigger weighted sum, then it willing to saturate in activation.
But in ReLU activation, it statistically cut half of it's input, so variance halves in each layer during forware path. **He et al, 2015** solve this problem by scaling up initial value.
```python
W = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in/2)
```
## Batch Normalization
It's good to make inputs of activation be normalized (unit gaussian), because constraint in input acts as a form of **regularization**. Therefore we insert **batch normalization layer** between fully connected part and activation part for each layers in network. Several advantages followed after batch normalization. 

```python
...
# Assume two layer neural net
# input size = N
# input dimension = D
# # of neurons in hidden state = H
# # of classes = C 
def loss(self, X_mini, y):
	W1 = self.params['W1'] # (D, H)
	gamma1 = self.params['gamma1']  # (1, H)
	beta1 = self.params['beta1'] # (1, H)
	W2 = self.params['W2']
	
	unnorm_hidden_state = X_mini.dot(W1) # unnormalized hidden state
	
	# batch normalization
	mean = np.mean(unnorm_hidden_state, axis=0)
	var = np.var(unnorm_hidden_state, axis=0)
	norm_hidden_state = (unnorm_hidden_state - mean) / np.sqrt(var + epsilon) # noramlized hidden state
	hidden_state = norm_hidden_state * gamma1 + beta1

	# activate
	hidden_state = ReLU(hidden_state)

	# get loss
	scores = hidden_state.dot(W2)
	loss = softmax(scores, y)
```

**Gradient flow.** Batch normalization spread input data before activation. So it makes gradient of activation more meaningful. Therefore, gradient flow through network is enhanced.  
**Learning rate.** As batch normalization normalize each layer's input to activation function, it preserves the gradient magnitudes during backpropagation. So, even though we choose higher learning rates, batch normalization prevents explosion of our model.  
**Dependence on initialization.** When we talked about weight initialization, we focused on way to maintain activations' distribution of each layers. Batch normalization explictly try to maintain inputs to activation in unit gaussian. Therefore, we can reduce dependency on initialization.  
**Act as a regularization.** In (pure) mini-batch optimization, each loss term only depends on single traning example during summation. But if we adapt batch normalization, each loss term depends on whole mini-batch training example by normalization process.  

### Note
**Test time.** In test time, we want deterministic batch-normalization layers. We have two methods. One is computing trainging data set's average and std after training finished. The other is estimating running average and std during traing process. Then we use those average and std in test time.  

## Babysitting the learning process
After preprocessing data and choosing network architecture, it is better to take **sanity checking** process.
* Without regularization, assuming softmax classifier and near-zero weight initialization, first value of loss should close to `ln(# of classes)`. It can be our first checkpoint.
* With regularization, first value of loss should increase.
* With small amount of training set, your network is possible to overfitting those data. We can check overfitting when training accuracy approaches 100%.
* If loss function decreases too slowly, increase learning rate then check whether loss function decreases faster.
* With very high learning rate, loss should explode.

## Hyperparameter optimization
* For double-type hyperparameters, optimize in log space.
* Start cross validation with coarse range, choose candidates, then proceed with fine range.
* If you have multiple hyperparameters, do not use grid layout. Optimal point can be placed between grids. 
* If your optimal point place in edges of interval, do not stop and search aside.
* If your training accuracy and validation accuracy differ too much, it means your network is overfitting traing data. So increase your regularization force or decrease your model capacity.
* Tracking the weight update to weight magnitude ratio. Somewhere around 0.001 is recommended. 

## Summary
**activation functions.** use ReLU  
**data preprocessing.** subtract mean image.  
**weight initialization.** use Xavier initialization.  
**batch normalization.** use it!  
**babysittig the learning process.** do it!  
**hyperparameter optimization.** random sample hyperparameters in log space when appropriate.