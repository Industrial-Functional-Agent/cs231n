# Backpropagation, Nerual Networks 1
## Gradient
In lecture 3, we talked about optimization. To opitmiaze parameters, we need (analytic) **gradient** of loss function. In neural network we used **computational graph** instead of one giant expression to compute gradient.

## Computational Graph
Computaional graph is composed of many kinds of **computational gate** which should provide **forward** and **backward** API. 
- **forward** : take inputs, compute output, then pass it to latter node.
- **backward** : take **global gradients** of latter node, compute global gradients of current node, then pass it to former node. 

Here is a example code of computation gate, multiply gate.
```python
class MultiplyGate(object):
	def forward(x, y):
		z = x*y
		return z
	def backward(dz):
		dx = y*dz
		dy = x*dz
		return [dx, dy]
```

## Backpropagation
In optimization process, our goal find optimial parameters minimizing loss function. When we use gradient descent technique in optimization, in each step, we need gradients of loss function. If we have **analytic expression** of loss function for parameters, we can compute gradients directly, but it's inefficient and complex. 
In computational graph, we have alternative called **backpropagation**, which compute gradients step by step. Each step takes global gradients of latter layer, chaining local gradients, then finally get global gradients of current layer. 

For effciency, we compute and save **local gradients** of each computational node in forward path. Then we can use in backward path.

## Vectorized operations
In practice, we deal with **computational layer** composed of many nodes to take advantage of vectorized operation in GPU. Here is example of vectorized operation (dot product). Latter one is much faster than former one.
```python
v1 = np.random.random((1, N))
v2 = np.random.random((1, N))
dot = np.zeros((1,N))

# dot product with iteration
for i in xrange(N):
	dot[i] = v1[i] * v2[i]

# dot product with vectorized operation
dot[:] = v1.dot(v2)
```
Therefore, input/output are no longer scalars, they are tensors. So local gradients are also (**Jacobian**) tensors. Theoretically, local gradient tensor's size is a combination of input tensor's size and output tensor's size. We can reduce it using some intuition. For example, in **max gate** local gradient tensor is actually diagonal, so we can store diagonal tensor instead of entire tensor.

## Lego block
In practice, we don't build entire neural network from scracth. Instead, our library (TensorFlow, Torch, Caffe, etc) provides many kinds of **unit layer**, then we build our network by **stacking** those unit layers in proper order, similar to lego block.

## Neural Network
Neural network is a computational graph which resemble human brain. Each neuron is connected with other neurons with **synapse**. A neuron takes other neuron's impulse from dendrites, compute its **activation** in cell body, pass it through axon to other neurons. So neural network is stacked by many layers of neurons.

To abstract human neuron, we simplify **tick** in two steps, **weighted sum** and **activation**.

![abstract human neuron](http://cs231n.github.io/assets/nn1/neuron_model.jpeg)

```python
class Neuron:
	def neuron_tick(inputs):
		cell_body_sum = np.sum(inputs * self.weights) + self.bias
		firing_rate = 1.0/(1.0 + math.exp(-cell_body_sum))
		return firing_rate
```