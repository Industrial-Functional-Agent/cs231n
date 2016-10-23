# Training Neural Networks, Part 2
* Parameter Updates
* Model Ensemble
* Dropout
* CNN: Brief History

### Question about last lecture
If you don't use a non-linear activation function, your neural network is just sandwich of linear function, those capacity is equal to that of just a linear classifier. 

## Parameter Updates
### SGD update
```python
while True:
	data_batch = dataset.sample_data_batch()
	loss = network.forward(data_batch)
	dx = network.backword()
	x += - learning_rate * dx
```
#### Why SGD is so slow?
Suppose loss function is steep vertically but shallow horizontally.
##### Q: What is the trajectory along which we converge towards the minimum with SGD?
Very slow progess along flat direction, jitter along steep one. Way too fast vertical direction, but way too slow horizontal direction.

### Momentum update
```python
# Momentum update
v = mu * v - learning_rate * dx # integrate velocity
x += v # integrate position
```
Loss function is a potential energy. The applying force to object is parallel to gradient of potential energy. The acceleration of object is sum of **friction** and **applied force**. Parameter updates with this acceleration. That is **momentum update**. 

* Physical interpretation as ball rolling down the loss function + friction(mu coefficient).
* mu = usually ~0.5, 0.9 or 0.99 (Somethime annealed over time, e.g. from 0.5 -> 0.99)
* If there is no mu(mu=1), the ball would never come to arrest because it was just jiggle around the loss surface forever. And there are will be no loss of energy where we'll settle at the end of a loss function.
* In shallow direction, the momentum update speeding up across the shallow direction.
* In steep direction, being pulled up the other direction toward the center and with the damping you end up kind of oscillating to the middle.

#### SGD vs Momentum
Even though momentum update overshoot the target, overall getting to the minimum is much faster.

### Nesterov Momentum Update
Compare to momentum update, Nesterov momentum update calculate gradients after updating parameter by amount of `mu * v`, which is called **lookahead gradient step**. It converges faster theoretically, also practically.  
```python
# Neterov Momentum update
v = mu * v - epsilon * gradient(x + mu * v)
x += v 
```
Slightly inconvenient.

```python
# Neserov Momentum update rewrite
v_prev = v
v = mu * v_prev - learning_rate * dx
x += -mu * v_prev + (1 + mu) * v
```

Local minimum is not issue. There is no bad local minima in big neural network.

### AdaGrad update
It is primarily introduced in convex optimization domain.
```python
# Adagrad update
cache += dx**2 # second moment
x += - learning_rate * dx / (np.sqrt(cache) + 1e-7)
```
* Added element-wise scaling of the gradient based on the historical sum of squared in each 
* In steep direction, the larger number is divided, so the update will be small.
* In shallow direction, the smaller number is divided, so the update will be large.

#### Q. What happens to the step size over long time?
* learning rate will decay to zero. 
* General convex optimal, it's okay. But in neural network it has a problem. It proves too aggresive and stops learning rate too early.

#### Problem
As cache monotonically increase, effective learning rate monotonically decrease. 

### RMSProp update 
It solves the problem of AdaGrad update. It is primarily introduced in a slide in **Geoff Hinton**'s Coursera course. It combines Adagrad update with concept of momentum update.
```python
# RMSProp
cahce = decay_rate * cache + (1 - decay_rate) * dx**2
x += learning_rate * dx / (np.sqrt(cache) + 1e-7)
```
#### Adagrad vs RMSProp
Adagrad is faster than RMSProp. But Adagrad converges too fast because its learning rate monotonically decreases. RMSProp converges in better point.
 
### Adam update
Adam update is most popular update rule. 
```python
# Adam
dx = # ... evaluate gradient
m = beta1*m + (1-beta1)*dx # update first momentum
v = beta2*v + (1-beta2)*(dx**2) # update second momentum
mb = m/(1-beta1**t) # bias correction
vb = v/(1-beta2**t) # bias correction
x += - learning_rate * m / (np.sqrt(v) + 1e-7)
```
* Looks a bit like RMSProp with momentum.
* beta1 and beta2 are hyperparameter. Pracitcal value for beta1 is **0.9**, and for beta2 is **0.995**.
* Bias correction step is for warming up `m` and `v`, to guarantee proper level of values in initial step.

### Learning Rate
All update techniques above has **learning rate** as their hyperparameter. Then what is **"good"** learning rate? We don't have general answer, but one obvious statement is that any fixed learning rate can't be a good hyperparameter for whole learning process. Start with higher learning rate, then decrease it during iterations. So we need good policy of decreasing learning rate.
#### Step decay
Step decay technique halves learning rate in every few epochs.
#### Exponential decay
Exponential decay decreases learning rate exponentially in every iteration.
#### 1/t decay
In 1/t decay, learning rate is inverse propotional to number of iterations.

All these three techniques came from domain of **convex optimization**. Even though we are in different domain, we can collaborate this learning rate decay into parameter update technique, **except Adagrad** update which embed monotonic decrease of learning rate.
## Second order optimization methods
Original second order optimization method deals with **Hessian matrix**. It starts with some point, derives second-order Taylor expansion, and solve for the critical point. Then we obtain the Newton parameter update.
```python
# derive critical point from starting point
critical = origin - inverse(Hessian) * gradient(origin) 
```  
Its advantage is that it doesn't need learning rate. It jump to critical point directly, then find next critical point and so on. But to compute critical point, we need to calculate **inverse matrix of Hessian**, which is almost impossible in practice.
### Quasi-Newton methods
#### BGFS
Specifically, BGFS method is most popular among quasi-newton methods. Instead of exact calculation of inverse Hessian with **O(n^3)** complexity, it approximates inverse Hessian with rank-1 update with **O(n^2)** complexity.
#### L-BGFS
`L` is named for **limited memory**. It doesn't store inverse Hessian in memory. I don't know detail of L-BGFS. Andrej said that it can work well in full-batch and deterministic mode. Adapting L-BGFS in large-scale and stochastic setting is still an active area of research.
### Summary
Use **Adam** update in default. Consider **L-BFGS** when full batch is possible.

## Model Ensemble
Basic concept of model ensemble composed of two steps. First, train **multiple** independent models. Second, **average** their result at test time. Practically, it achieves **2%** better performance. There are many different model ensemble techniques in practice.
### Checkpoints
If training costs a lot, we can use checkpoints during single learning process in model ensemble. Practically, it also achieves better performance.
### Running Average
This technique additionally calculate running average of parameters in training process, then use running average instead of final parameter at test time. 
```python
x += - learning_rate * dx # simple SGD
x_test = 0.995 * x_test + 0.005 * x
```  
Running average can be effective in situation like parameters rotating around exact minimum in latter part of training. Even though parameters rotating around in each iteration, their running average can direct exact point of minimum.

## Dropout
In forward propagation, mask few randomly chosen neurons to zero.
```python
p = 0.5 # probability of keeping a unit active. higher = less dropout

def train_step(X):
	""" X contains the data """

	# forward pass for example 3-layer neural network
	H1 = np.maximum(0, np.dot(W1, X) + b1) # ReLU activation
	U1 = (np.random.rand(*H1.shape) < p) / p # first dropout mask
	H1 *= U1 # drop!
	H2 = np.maximum(0, np.dot(W2, H1) + b2)
	U2 = (np.random.rand(*H2.shape) < p) / p # second dropout mask
	H2 *= U2 # drop!
	out = np.dot(W3, H2) + b3

	# backward pass: compute gradients... (not shown)
	# perform parameter update... (not shown)

def predict(X):
	# ensembled forward pass
	H1 = np.maximum(0, np.dot(W1, X) + b1) # NOTE: scale the activations
	H2 = np.maximum(0, np.dot(W2, H1) + b2) # NOTE: scale the activations
	out = np.dot(W3, H2) + b3
```
### As model ensemble
Dropout can be interpreted as model ensemble with shared parameters. Each binary mask corresponds to single model. But each single model is trained on only one training data.
### Redundant representation
Dropout can be interpreted as forcing the neural network to have a redundant representation. Without dropout, evaluation depends on whole neurons. With dropout, few neurons are masked out and evaluation only depends on remaining neurons. Therefore network is trained to evaluate well even on the situation that only portion of neurons are used.

## CNN: Brief History
### Hubel & Wiesel (1959)
They researched visual cortex of cat. And found out two keypoints. **First**, some neurons focus on some orientation of visual stimulus. **Second**, adjacent region in vision also activates adjacent region in visual cortex. They concluded this results in **hierarchical organization** of human vision.
```
simple cell ---> complex cell ---> hyper-complex cell
```
### Fukushima (1980)
Based on results of Hubel & Wiesel, He introduced concept of **Neurocognitron**. He construct hierarchical network with simple cells and complex cell. But he didn't adapt back propagation.
### LeCun et al (1998)
They first combinate hierarchical architecture with technique of back propagation.
### AlexNet (2012)
Huge success in computer vision area with deep neural network. Even though big difference in performance, architecutre of AlexNet does not differ much with LeCun's network. ReLU instead of tanh activation, bigger/deeper network supported by GPU machine, and a lot of data set.