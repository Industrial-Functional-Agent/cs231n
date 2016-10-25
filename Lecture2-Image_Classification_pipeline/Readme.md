# Image Classification pipeline
[python/numpy tutorial](http://cs231n.github.io/python-numpy-tutorial/)  
[image classificatoin notes](http://cs231n.github.io/classification/)  
[linear classification notes](http://cs231n.github.io/linear-classify/)

# Image Classification
## A core task in Computer Vision.  
**input** : image (보통 3d array of image)   
**output** : label (given set of discrete lables 중)

## Problem
**semantic gap** : 사람은 이미지를 이미지 그 자체로 인식하지만, 컴퓨터는 3차원 배열 데이터로 인식한다.

## Challenges
* **viewpoint variation** : 같은 사물도 다른 방향에서 바라보면 전혀 다른 3차원 배열 데이터가 된다.
* **illumination** : 조도를 바꿔도 마찬가지다.
* **deformation** : 고양이가 다양한 자세로 누워있을 수 있다.
* **occlusion** : 일부분만 보일 수 있다.
* **background clutter** : 배경과 사물이 유사한 경우가 있다.
* **intraclass variation** : 같은 고양이여도 다양한 종이 있다.

## hard-coding algorthm vs data-driven approach
* **hard-coding algorithm** : 고양이를 인식하는 알고리즘을 만들어내는 명확한 방법은 없다.
* **data-driven approach** : 데이터로 **model**을 만들고, **model**로 **predict** 한다.
	* **collect** : image/label 들을 수집한다. 
	* **train** : machine learning 을 활용해서 image classifier를 train 한다.
	* **evaluate** : test image 들로 classifier를 평가한다.

## Nearest Neighbor Classifier
* **train** : Remember all images & labels
* **evaluate** : 새로운 test image가 들어오면, 가장 가까운 image의 label로 labeling 한다.
* **performance** : classification speed가 training data 크기에 **비례**한다.
	* training 시간은 짧고, evaluation 시간이 길다.
	* 중요한건 **test time performance**이므로, NN classifer는 나쁘다.
	* **그래서 image에는 절대 안쓴다.**
	* 이에 반해 neural network는 반대 경향성을 보인다. 그래서 좋다.
* 일반적으로 **k-NN classifier**로 일반화된다.
* **Q.** k-NN classifier에서 k는 어떤 의미를 지니는가? 
 
## Hyperparameter
* **parameter** : (deep learning의 경우에는) `weight` 와 `bias`를 의미한다. 즉, **learning** 과정에서 **update** 되어 **convergence**에 이르는 것들을 총칭.  
* **hyperparameter** : **learning model**의 **parameter**를 뜻한다.
	* loss function의 선택
	* (k-NN classifer의 경우) k
	* (gradient descent의 경우) learning rate
	* (deep neural network의 경우) layer의 층수
* **Q. how do we set the hyperparameters?**
	* 딱히 문제 의존적이다. 명확히 정해진 답이 없다.
	* 한가지 방법론이 **cross validation**이다.
	* **주의** : **test data**로 hyperparameter를 결정하면 안된다!!!
	* **이유** : **over-fitting**      

## Cross Validation
1. **training data**를 **N**개의 **fold**로 나눈다 (각각이 validation fold가 된다)
2. 하나의 **fold**를 선택한다.
3. 나머지 N-1개의 fold로 model을 **learning** 한다.
4. 2에서 선택한 fold로 3의 model을 **evaluate** 한다.
5. N개의 fold 각각에 대해서 2-4의 과정을 반복한다.
6. N개의 **결과**와, **평균**과, **표준편차**를 계산한다.
7. 6에서 **평균 성능이 가장 좋은** hyperparameter를 선택한다.

## Linear Classification
* **Lego block** : 최신 Google Net 같은 복잡한 네트워크도 linear classifier 같은 간단한 레이어의 중첩으로 이뤄져있다. 
* **자랑** : Andrej Karpathy는 `cnn`뿐 아니라 `rnn`에도 상당한 고수다.

### parametric approach
* 32x32x3 array + parameters(, or weights) ---> 10 numbers

### Q1. what does the linear classifier do, in English?
* 스탠포드 학생들의 다채로운 답변들이 있었다.
* **Andrej가 가장 먼저 소개한 답변** : 이미지의 색깔 정보의 가중치 합을 계산한다.
* classifier들을 다시 image 모양으로 reshaping 하고나면 **template image** 처럼 보인다.
* 또한 각각의 classifier는 class를 나누는 (고차원 공간에서의) **hyperplane** 역할을 한다.

### Q2. what would be a very hard set of classes for a linear classifer to distinguish
1. **gray scale image** : linear classifier의 색깔 정보 가중치합이 제대로 동작하지 못한다. 
	* **why?** : 모든 image들이 비슷한 결과값을 내기 때문이다.
2. **large intraclass color variance** : 일관성이 없어서 제대로 된 **template**이 나올 수 없다.

## Coming up....!!
* **loss function** : 좋은 parameter를 정의하는 기준 (quantification)
* **optimization** : 좋은 parameter를 찾아나가는 방법
* **ConvNets** : 좋은 model
