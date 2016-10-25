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