## Done
- [x] Affine layer: forward
- [x] Affine layer: backward
- [x] ReLU layer: forward
- [x] ReLU layer: backward

## Modularization
```python
def layer_forward(x, w):
	...
	return out, cache
```
반환된 **cache**는 backward prop 과정에서 쓰인다. 
```python
def layer_backward(dout, cache):
	...
	return dx, dw
```
## "Sandwich" layers
자주 쓰이는 패턴들(예: Affine layer + ReLU layer)을 미리 합쳐두고 사용할 수 있다. 일종의 캡슐화라고 생각했다.
## Loss layers: Softmax and SVM
**Assignment 1**에서 구현했던 내용이므로 코드가 주어졌다. 내 구현과 비교하면 좋을 듯 하다.
## Solver
뉴럴 네트워크의 구현체와, 최적화하는 객체(=solver)를 분리한다. Assignment 1에서는 합쳐져 있었다. learning rate decay는 매 epoch 마다 이뤄진다. 다양한 hyperparameter 들을 쫙 리스트업 하고 어떠한 경우에 크게/작게 변화시킬지 잘 정리해두면 좋을 것 같다(Andrew Ng 강좌에서 잘 소개해줬던 기억이 난다).
- reg (model)
- print_every (solver)
- learning_rate (solver) 

당장에 reg 랑 learning_rate 만 적절히 조절해서 목표로 했던 val accuracy 50% 를 달성할 수 있었다. epoch 에 대한 accuracy 그래프를 보고 수렴이 될 된것 같으면 learning_rate 를 키우고, 중간에 loss 가 nan 이 뜨면 learning_rate 를 낮춘다. 그리고 training 과 validation 의 차이가 커보인다면 reg 값을 키운다.
## Dropout
randomly setting some features to zero during the **forward pass**.
implement a dropout layer and modify fully-conneted network to optionally use dropout.

dropout 은 training/testing 에 따른 operation 이 다르므로 이에 유의한다.

dropout forward/backward 코드를 작성한다. 구현해야할 것은 구체적으로 **"inverted dropout"** 이다.
dropout 시킨만큼, 남아있는 neuron output 들의 값을 scale-up 해서 평균치를 맞추는 기법이 inverted 였다. (p로 나눈다고 inverted)
구현은 간단했다.

구현한 dropout layer 들을 fc_net 에 적용한다.

적용하고 drop-out 여부에 따른 model 의 training 을 확인해본다.
validation accuracy 는 비슷한데, training accuracy 가 좀 줄었다. drop-out 없을때는 거의 1을 찍는다.
그것이 바로 **regularization** 이다.
## ConvNet
### conv_forward_naive
* numpy.empty
* numpy.pad
* numpy.flatten
* numpy.inner

CONV layer 에 ReLU 를 기본으로 깔고 들어가는지 알았는데, conv_forward_naive 에서 ReLU 는 포함시키지 않는다. 이미 ReLU layer 를 구현해놨으니, integrate 하는 단에서 이를 합성하려나 싶다.
### Aside: image processing by convolution
간단한 문법으로 image processing 을 할 수 있었다.
### conv_backward_naive
dx 를 하니까, dw 를 하는건 쉬웠다.
올? 한번에 통과했네?
### max_pool_forward_naive
* numpy.amax

### max_pool_backward_naive
* numpy.argmax
* numpy.unravel_index

max pooling 하는 mask 에서 maximum 인 point 가 여러개 있는 경우는 커버하지 못한다. numpy.argmax 함수가 첫번째 maximum 의 index 만 반환하기 때문.
### Fast Layers
Jupyter 가이드를 따라 setup.py 를 실행하면, 에러가 발생한다. 
```cmd
running build_ext
building 'im2col_cython' extension
error: Unable to find vcvarsall.bat
``` 
http://stackoverflow.com/questions/2817869/error-unable-to-find-vcvarsall-bat
이 링크의 방법을 시도해보니 해결되었다. 나는 14 버전의 Path 가 설정되어있어서 그 값으로 지정했다.
### ThreeLayerConvNet
#### implementation
* **stride** / **pad** 값은 CONV layer 이후 데이터의 크기를 유지하도록 설정되어있다.
* conv_relu_pool_forward/backward 이전/이후 에서는 데이터를 **reshape** 해야하는 이슈가 있었다.

#### validation
* loss/gradient 값 체크
* 작은 크기 데이터의 over-fitting 가능 여부 체크
	* training accuracy 는 거의 1에 가깝게 나온다.
	* validation accuracy 는 큰 폭으로 작다.

### Spatial Batch Normalization
* batch = N개 데이터에 대한 statistics 를 내겠다.
* spatial = H x W 공간에 대한 statistics 를 내겠다.

두개를 합쳐서 이름이 **spatial batch normalization** 인가보다.

Batch Normalization 을 사용해서 짧게 구현할 수 있다. 적절히 **transpose** 하고 **reshape** 해주는 작업을 왕복해주면 된다. forward/backward 마찬가지다.