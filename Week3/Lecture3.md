# Linear Classification 2, Optimization

## ToDo:
1. Define a **loss function** that quantifies our unhappiness with the scores across the training data.
2. Come up with a way of efficiently finding the parameters that minimize the loss function. **(optimization)**

## Multiclass SVM loss:
Here is a problem for us. We can write math equation...
What can we do?

+1 is a safety margin = not a hyperparameter

## Q1: What if the sum was instead over all classes? (including j = y_i)
* Answer: 모든 Loss에 1씩 추가 될 것이므로, 전체 적인 로스가 증가 할 것이다. 그러나 Optimal W는 동일

## Q2: What if we used a mean instead of a sum here?
* Answer: 전체적인  Loss는 감소하지만, optimization입장에서는 큰 차이 없을 것이다.

## Q3: L2 Loss를 쓴다면?
* Answer: Different W를 얻는다. scailing할때 constant만큼 scaling되지도, shifting되지도 않았다. 로스가 큰놈은 훨씬더 로스가 크게 된다. 로스가 큰 애들에게 민감하게 optimization될 것 이다.

## Q4: What is the min/max possible loss?
* Answer: min = 0, max = infinite

## Q5: Usually at initialization W are small numbers, so all s ~= 0. What is the loss?
* Answer: number of classes - 1 => sanity check
	* 만약 W를 적게 해서 첫번째 loss를 구했다면 number of classes - 1정도가 나와야 한다.

## Suppose that we found a W such that L = 0. Is this W unique?
* Answer: No, We must regularize the W: entire subspace of W

## Weight Regularization
lambda = regularization strength(hyperparameter)

## L2 regularization
* reason = spread out w as much as possible that you're taking into account all the input features. Since the L2 penalty prefers smaller and more diffuse weight vectors, the final classifier is encouraged to take into account all input dimensions to small amounts rather than a few input dimensions and very strongly.

## Softmax Classifier(Multinomial Logistic Regression)
### Q: What is the min/max possible loss L_i?
* Answer 0 and infinite	