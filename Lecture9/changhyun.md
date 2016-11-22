[pretrained ResNet 의 GitHub repository](https://github.com/KaimingHe/deep-residual-networks)
(152-layer ILSVRC 2015 winning ConvNets)

* Visualize patches that maximally activate neurons
* Visualize the weights
* Visualize the representation space (e.g. with t-SNE)
* Occlusion experiments
* Human experiment comparisons
* Deconv approaches (single backward pass)
* Optimization over image approaches (optimization)

ConvNet 상에서 어떤 layer 를 잡고, 이미지들을 쏟아부으면서, neuron 을 가장 activate 시키는 이미지들을 뽑는다. 그리고 그 이미지들이 실제로 무엇인지 본다.

이를 통해 사람의 상반신을 찾는 neuron, 강아지를 찾는 neuron 등등을 파악할 수 있다.

## Visualize the filters/kernels
첫번째 CONV layer 에 대해서만 가능하다. 유일하게 raw image 에 가해지는 경우이기 때문.

어떤 알고리즘을 쓰던, 어떤 raw image 에 가해지던, 결국 training 후에 첫번째 CONV layer 에는 gabor-like filter 가 만들어진다는 얘기가 있었다.

## Visualize the representation
classifier 직전 layer 의 값을 그 이미지의 "code" 로 생각할 수 있다. 이를 **representation** 이라고 하나봄.
### t-SNE visualization (van der Maaten & Hinton)
기본적인 원리는 high-dimensional points 들을 2-dimensional 로 **embedding** 하는 것이다. 이때 high-dimensional poi nts 들 사이의 pair-wise distance 를 보존한다.

Karpathy 는 이를 ImageNet Challenge 에 쓰인 image 들의 CNN "code" 에 적용한 그림을 보여줬다.
## Occlusion Experiments
원본 image 가 있고, true label 이 주어진다. 작은 zero-patch 를 가지고 원본 image 를 sliding 하면서 (CNN) classifier 가 true label 를 선택할 확률을 heat map 으로 그린다.

이를 통해 true label 로 분류하기 위해 중요한 부분 / 방해되는 부분을 인지할 수 있다.

## Visualizing Activations
Jason Yosinski 의 비디오를 보여줬다. Real-Time 으로 계산되는 CNN 의 각 레이어별 activation 을 웹에서 관찰할 수 있다.
### Deconv approaches
어떤 neuron 의 activation 에 대한 특정 image 의 gradient 를 구하려면?

* 해당 neuron 이 존재하는 layer 까지 forward propagate
* 해당 neuron 이 존재하는 layer 의 local gradient 를, 해당 neuron 의 값만 1로 하고 나머지는 모두 0으로 설정
* 그 layer 에서부터 backward propagate
* (image 와 같은 크기의) gradient 획득

**guided backpropagation**

ReLU 의 경우 본래 backpropagation 에서는 forward 에서 0 이상이었던 값들의 gradient 만 패스시키는 역할이었다. 이에 반해 guided backpropagation 에서는 추가로 음수인 gradient 값도 차단한다.

이를 통해서 목표로 하는 특정 neuron 에 positive 영향을 끼치는 backpropagation 경로만 고려된다. positive 와 negative 의 싸움이 아니라, positive 경로만 고려되므로 특정 neuron 에 대한 input image 의 gradient 를 더 깔끔한 map 으로 얻을 수 있다.

**backward deconvnet** 

이 경우는 forward 에서 0 미만의 값들의 gradient 를 차단시켰던걸, 풀어준다.

Deconv approaches 는 forward/backward 한번이면 gradient 를 해킹해서 무언가 시각화를 할 수 있으므로 연산이 저렴하다 할 수 있겠다. 

### Optimization approaches
질문: Can we find an image that maximizes some class score? (regularization term 도 들어간다. loss 에서는 +로 들어갔으니, 여기서는 maximize 하므로 -로 들어간다)

1) zero image 를 넣는다
2) 원하는 score 까지 forward
3) (Deconv approaches 와 비슷한 테크닉) score 를 표현하는 텐서에서 원하는 score 만 1로 두고, 나머지는 모두 0으로 둬서 가장 뒷단의 gradient 를 만든다.
4) input image 까지 backward
5) input image update
6) 반복....!!

(갑자기 다른 맥락으로 넘어간다)

어떤 라벨의 이미지를 넣고, 해당 라벨의 score 에 대한 gradient 를 위의 프로세스로 구할 수 있다. 이 경우 gradient 는 각 pixel 이 해당 이미지가 해당 라벨로 분류될 수 있도록 기여하는 정도의 표현이라고 볼 수 있다. 예를 들어 강아지 이미지를 넣고 강아지 라벨에 대해서 gradient 를 얻고 이를 그려보면, 강아지의 얼굴이나 몸체 부분의 값이 클 것이다.

이를 기반해서 **grabcut** 을 써서 segmentation 한다.

질문: Given a CNN code, is it possible to reconstruct the original image?
(code = classifier 들어가기 직전의 tensor 값) 

* its code is similar to a given code
* it "looks natural" (image prior regularization)

## Deep Dream
GitHub 들어가서보면 코드가 100여줄 정도다. 간결하다.
```python
def make_step(...):
	# image 에 대한 메소드
	# 한번 호출될때마다 입력으로 받은 image 를 조금 업데이트한다 (optimize 하는 방향으로)
```
실제로 `make_step` 메소드가 입력으로 받는건 image 가 아니라 CNN 에서 어느 layer 를 "dream" 할 것인지에 대한 identifier 이다.

* 어느 layer 까지 forward
* 그 layer 의 activation 을 정확히 시작 gradient 로 지정
* 맨 앞까지 backward

말로 표현하자면 active 한 activation 을 더 activate 하게 boost 하는 것이 위의 작업이라고 한다. 정확히 이해를 못했다. 예를 들어 평범한 "sky" 이미지를 줘도, 중간 layer 의 activation 을 봤을때 "dog" 을 분류하는 classifier 의 activation 이 어느정도 값일 수는 있다. 구름이 약간 얼굴처럼 생길 수 있기 때문이다. 여기에 dream 을 계속 가하면, 그 activation 이 점차 커지는 방향으로 boost 되므로 이미지의 해당 부분에 개의 형상이 나타나는 것이다.

슈퍼에서의 영상을 프레임 단위로 Deep Dream 하고 Optical Flow 로 smoothing 한 동영상이 아주 인상적이다.

## Neural Style
content image 와 style image 로 나뉜다.

**step 1**
content image 를 CNN 에 넣어서 모든 layer 의 activation 을 획득한다. 이를 `content targets` 라 한다.

**step 2**
target image 를 마찬가지로 CNN 에 넣어서 모든 layer 의 activation 을 획득한다. 하지만 이를 그대로 사용하지 않는다. 예를 들어 activation map 들이 적층된 tensor 가 `224x224x64` 의 크기라고 하자. 그럼 각각의 `1x1x64` 를 **fiber** 라 부른다. 그럼 총 `224x224` 개의 fiber 들이 있는 것이다.

이제 모든 fiber 의 pair 에 대해서 outer product 로 `64x64` 를 계산하고 더해서 최종 `64x64` 의 tensor 를 만들어낸다. 이를 **Gram Matrix**(by paper) 혹은 **Covariance**(본래 statistics 의 개념?) 라 부른다.

이를 간단하게 계산하는 방법은, fiber 들을 모두 이어붙여서 row 가 `224x224`, column 이 `64` 인 행렬을 만든 뒤 (=V) V^T x V 하는 것이다.

**step 3**
이제 optimization 이다. objective 는 content target 과 gram matrix 와의 distnace(?) 이다. 전체를 다 비교하는건 아니고, 각각에서 특정 layer 에 대한 것만 골라서 objective 로 삼는듯.

Johnson 의 코멘트에 따르면, 꼭 gram matrix 여야하는건 아니다. 저 자리에 들어가야할 것은 style image 로 뽑아낸 어떤 invariant 한 statistic information 이다. 더 간단하게는 feature map 의 average 같은 것도 쓸 수 있는 것이다.

Karpathy 의 코멘트에 따르면, L-BFGS 로 가장 업데이트가 잘 된다. 이미지 1장에 대한 작업이므로 모든것이 메모리에 fit 한다. 그러니 2nd order optimization 을 사용할 수 있다는 듯.
## Adversarial Example
앞의 내용에서 어떤 input image 에 대해 네트워크 상의 임의의 neuron 의 값을 optimize 하는 테크닉이 많이 등장했다. 이제 이에 익숙해졌다.

그러니 예를 들어서 임의의 input image 에 대해서 ConvNet 의 classifier 의 ostrich(타조) score 를 optimize 하는 방향으로 image 를 update 할 수 있다. 우리의 예상은, 이를 통해 image 가 ostrich 에 가까운 image 로 변화할 것이라는 점이었다. 

하지만 결과는 예상과 매우 달랐다. 사람이 보기에 그 변화를 느끼기 힘들 정도로 자잘한 distortion 을 가하는 것만으로도 ostrich score 가 매우 높아지도록 만들 수 있었다. 같은 작업을 꼭 image 에서 시작하지 않고 완전히 random noise 에서 시작할 수도 있었다. 결과는 마찬가지였다. random noise 에서 다른 randon noise 로 업데이트해서 특정 class 의 score 를 높힐 수 있었다. 무려 **99.6%** 이상의 confidence 였다. 

이런 현상은 ConvNet 이전부터 있어왔다. HOG 를 예로 들면, 사람이 보기에 완전히 다른 image 인데 HOG feature 는 거의 동일한 경우를 만들 수 있었다.

이에 대한 해석을 실은 논문이 있다.
> primary cause of neural networks' vulnerability to adversarial perturbation is their **linear nature**.

매우 고차원 공간에서 아주 제한적인 경로 (manifold) 를 탐색해서 ConvNet 이 완성되고 나니, 미처 탐색하지 못한 지역(shadow)에서 엄청난 chaos 를 경험한다는 얘기다.

### fool a binary linear classifier
꽤나 고차원 벡터에 대한 binary linear classifier 를 생각한다. affine transform 하고나서 logistic regression 하면 class 0 에 대한 probability 가 된다.

우리가 weight 값을 알고 있는 상황에서 어떤 x가 주어졌을 때, 이 x와 유사하면서 class 0 에 대한 probability 값은 아주 차이나는 adversarial x 를 만들 수 있다.

예를 들어 본래 x는 class 0 에 대한 probability 가 매우 낮다고 가정하자. weight 를 알고 있으므로, 각 dimension 을 돌면서 해당 축의 dimension 값이 음수라면 adversarial x는 x의 해당 축 값보다 조금 자게, 양수라면 조금 크게 설정한다. 고차원 벡터이므로 x와 adversarial x는 상당히 가까운 데이터가 될 것이지만 매 축에서 affine transform 의 값이 커지므로 logistic regression 의 값이 원래는 0에 가까웠다가 1에 가까워질 수 있다. 

ConvNet 에 대해서도 마찬가지로 작용한 것이다. 비단 ConvNet 만의 문제가 아니라, 어떤 modality 의 NeuralNet 이라도 같은 문제를 지니고 있다. data manifold 전체를 탐색해서 learning 하는게 아니라, 실질적으로 linear nature 에 기반하므로 low-level statistics 를 가지고 learning 하는 셈이기 때문이라 한다.

### 몇가지 idea
* image 통째에 대해서 classify 하지 않고, 여러 crop 을 내서 classify 하고 그 결과를 수합한다.
* 본래 하듯이 network 를 training 하고, training image 에 대한 adversarial image 를 생성해낸 뒤, 얘네들을 본래 class 에 대한 negative class 에 속한다고 라벨링해서 더 training 시킨다. 근데 잘 동작하지는 않는다. 계속해서 더 많은 adversarial image 가 생산된다고 한다.

추가로, adversarial example 이 문제가 될 수 있는 이유는 세상에 여러 ConvNet 이 존재하기 때문이다. 그러므로 의도치 않게 내가 찍은 사진에 올바른 라벨을 매긴것이 어떤 ConvNet 에서 adversarial example 로 작용할 수 있다.