# Linear Classification 2, Optimization

## Loss function
Loss function is a function that quantifies our unhappiness with the scores across the training data.
### Multiclass SVM loss:
```python
def L_i_vectroized(x, y, W)
	scores = W.dot(x)
	margins = np.maximum(0, scores - scores[y] + 1)
	margins[y] = 0
	loss_i = np.sum(margins)
	return loss_i
```

+1 is a safety margin = not a hyperparameter.

#### Q1: What if we used a L2 loss?
* The choice of selections is another hyperparameter.
* The optimized W will be different.

#### Q2: Usually at initialization W are small numbers, so all s ~= 0. What is the loss?
* The loss is **number of class - 1**.
* Check point for initial learning step.

Unfotunately, there is a **bug** in above code. The optimal W is not unique. (What if we multiply **alpha > 1** to optimal W?) Because in above code, we search the optimal W in whole space of W. So we wish to encode some preference for a certain set of weights W over others to remove this ambiguity. 

### Regularization
What is a regularization? It is a representation of preference. And there are many kinds of preferences.

* L2
* L1
* Elastic(L1 + L2)
* Max norm
* Dropout

For example, L2 normalization prefers the distributed weights to skewed. Diffuese weights are good in somecase. Because the result is affected by **all features**. Let's assume that there are two weights which result in same loss.

1. [1, 0, 0, 0]
2. [0.25, 0.25, 0.25, 0.25]

L2 normalization favor the second one. 

### Softmax loss (Multinomial Logistic Regression):
In softmax loss, scores are normalized log proboblities of the classes. We want to maximize the log likelihood, or (for a loss function) to minimize the negative log likelihood of the correct class.

### Softmax vs. SVM
In practically the results W of two loss functions are usually comparable. 

*  Jiggle the data a bit (changing its score slightly). What happens to the loss in both cases? 
	* Softmax: The loss will change slightly.
	* SVM: The loss will not change. Because SVM has a robustness due to the margin +1.

## Optimization
Optimization is finding parameters that minimize the loss function. (efficiently)

### Gradient descent
```python
while(true)
	weights_grad = evaluate_gradient(loss_fun, data, wegihts)
	weights += -step_size * weights_grad # Perform parameter update
```

* numerical
	* easy to write
	* slow
	* very slow to evaluate

* analytic
	* fast
	* error prone
	* exact 	

In practice, we always use analytic gradient, but check implementation with numerical gradient. This is called a **gradient check**.

#### Mini-batch Gradient descent
* only use small portion of the training set to compute the gradient.

```python
while True:
	data_batch = sample_traging_data(data, 256) # samples 256 samples
	weights_grad = evaluate_gradient(loss_fun, data_batch, weigths)
	weights += -step_size * weights_grad # perform parameter update
```

Common minibatch sizes are 32/64/128 examples.

Before 2012, we did feature extraction hardly. But now, the neural network does it for us.