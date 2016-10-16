# Training Neural Networks, Part 2
* If you don't use a non-linear activation function, your neural network is just sandwich of linear function, those capacity is equal to that of just a linear classifier. 

## Parameter Updates
### SGD update
* Training a neural network, main loop:

	```python
	while True:
		data_batch = dataset.sample_data_batch()
		loss = network.forward(data_batch)
		dx = network.backword()
		x += - learning_rate * dx
	```
	
* Why SGD is so slow?
* Suppose loss function is steep vertically but shallow horizontally
	* Q: What is the trajectory along which we converge towards the minimum with SGD?
	* Very slow progess along flat direction, jitter along steep one. Way too fast vertical direction, but way too slow horizontal direction.

### Momentum update
```python
# Momentum update
v = mu * v - learning_rate * dx # integrate velocity
x += v # integrate position
```

Loss function is a potential energy. The applying force to object is parallel to gradient of potential energy. The acceleration of object is sum of friction and applied force. Parameter updates with this acceleration. That is momentum update. 

* Physical interpretation as ball rolling down the loss function + friction(mu coefficient).
* mu = usually ~0.5, 0.9 or 0.99 (Somethime annealed over time, e.g. from 0.5 -> 0.99)
* If there is no mu, the ball would never come to arrest because it was just jiggle around the loss surface forever. And there are will be no loss of energy where we'll settle at the end of a loss function.
* In shallow direction, the momentum update speeding up across the shallow direction
* In steep direction, being pulled up the other direction toward the center and with the damping you end up kind of oscillating to the middle.

### Neterov Momentum update
```python
# Neterov Momentum update
v = mu * v - epsilon * d(x + mu * v)
x += v 
```
Slightly inconvenient.

```python
# Neserov Momentum update rewrite
v_prev = v
v = mu * v -learning_rate * dx
x += -mu * v_prev + (1 + mu) * v
```

Local minimus is not issue. There is no bad local minima in big neural network.

### AdaGrad update
```python
# Adagrad update
cache += dx**2 # second moment
x += - learning_rate * dx / (np.sqrt(cache) + 1e-7)
```

* Added element-wise scaling of the gradient based on the historical sum of squared in each 
* In steep direction, the larger number is divided, so the update will be small.
* In shallow direction, the smaller number is divided, so the update will be large.
* Q2: What happens to the step size over long time?
	* learning rate will decay to zero. 
	* General convex optima, it's okay. But in neural network it has a problem. It proves too aggresive and stops learning rate early.

### RMSProp update 
```python
# RMSProp
cahce = decay_rate * cache + (1 - decay_rate) * dx**2
x += learning_rate * dx / (np.sqrt(cache) + 1e-7)
```

Combine Adagrad and Momentum
### Adam update
(incomplet, but close)

```python
# Adam
m = beta1*m + (1-beta1)*dx # update first momentum
v = beta2*v + (1-beta2)*(dx**2) # update second momentum
x += - learning_rate * m / (np.sqrt(v) + 1e-7)
```

* Looks a bit like RMSProp with momentum
* beta1 and beta2 are hyperparameter

compensate
