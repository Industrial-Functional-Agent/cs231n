# Backpropagation, Nerual Networks 1

## Computational Graph
Instead of one giant expression, we can visualize our complex computation using **computational graph**. Computaional graph is composed of many kinds of **computational gate** which accepts inputs, then compute, and deliver it to next gate.
Using this concept, we can represent neural networks as computational graph.  


Lecutre노트를 모두 다 읽어라,k

 그래디언트 디센트
 * Numerical Gradient: slow (why? w vector size is same with feature vector size. in convnet feature vector size is approximately million. So for one step update w, we neet do calculate gradient million times.) But it is easy to write.
 * Analytic gradient: fast (why? just some math equation? Actually, i don't get it why this analytic gradient is fast.), exact, but error-prone
 
 In practice: use analytic gradient, check your implementation by using numerical gradient
 
computational graph
if i increase z amount of h, then f increase 3 * h 
chain rule!

포워드나 백워드나 시간이 비슷하다?
backward is slightly slower

if we use sigmoid function(gate) and it's derivative, we can express more shortly. => it it more easy to compute?

add gate: gradient distributor
max gate: gradient router -> smaller one doesn't effect the loss so the backword flow only propagates to larger one
mul gate: gradient "switcher"? not good.

Gradients add at branches

There are never be loop

implementation: cache the data

layers = building block 
library is whole set of gate

every step: forward backward update

you don't actually need to ever form those jacobians

max function's Jacobian is diagonal, and it's element almost zero

Neural Networks
more compley f = W2max(0,W1x)

have template of all car

if h has less than 10 units, would be inferior to linear classifier?

size of hidden layer = hypermarameter

single neuron = linear classifier

first layer가 공간을 비틀고 second layer가 리니얼리 클래시파이 한다. => 커널 트릭?

wide/deep ?