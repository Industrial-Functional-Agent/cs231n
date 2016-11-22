- [x] pre-processing (RGB channel 빼주는 것. per-channel 이고 전체 data set 에 대한 mean 으로)
- [ ] small data 에 대해 over fitting 이 되는지 검증 (VGGNet 자체에 대한 validation)
- [ ] Xavier initialization
- [ ] batch normalization (or dropout)

## VGGNet Architecture
논문에 제시된 Table 1 에서 A column 의 세팅을 모방한다. 모든 POOL layer 는 2x2 max pooling 에 stride=2 세팅이다.

* CONV1 (64 filters) - POOL1
* CONV2 (128 filters) - POOL2
* CONV3 (256 filters) - CONV4 (256 filters) - POOL3
* CONV5 (512 filters) - CONV6 (512 filters) - POOL4
* CONV7 (512 filters) - CONV8 (512 filters) - POOL5
* FC (4096) - FC (4096) - FC (10) - SOFTMAX

선택할 것
* 첫번째/두번째 FC 크기. 동일하게 가져간다고 생각해볼까?

## Iteration Time
논문에서 FC layer 의 neuron 개수가 **4096개** 이다. VGGNet 에서 마지막 POOL5 후에 FC layer 로 집어넣기 전에 단면을 평균 내는 작업이 있다. 그러므로 FC layer 에 들어가는 input tensor 크기는 `1x1x512` 가 된다. 위의 architecutre 를 따른다면 CIFAR-10 에 대해서도 마찬가지다. `1x1x512` 라는 크기는 CONV8 layer 에서 사용한 filter 의 개수에 의해 결정되기 때문이다.

batch_size = 50 일때 `Solver` 인스턴스의 _step 메소드 호출 시간이 아래와 같다. _step 메소드를 호출하면 forward / backward 한번 계산한다.

FC layer size|iteration time(sec)|
-------------|-------------------|
4096         |7.446              |
256          |4.730              |

## Over-Fitting small data
`Solver` 인스턴스의 train 메소드를 호출해서 몇 iteration 을 진행해봐도 loss 가 줄지 않는다. 애초에 구현이 제대로 되었는지 의심되는 상황이다. 의문을 해소하기 위해서 작은 data set 을 떼내서 network 가 이를 over-fitting 시킬 수 있는지 확인한다.

잘 되지 않는다. 작은 data set 을 떼내서 돌려봐도 loss 가 줄지 않는다.
### initialization
아무래도 initialization 의 문제인듯 하다. Xavier initialization 을 쓴다.
### batch normalization
* conv-batch-relu-pool
* conv-batch-relu

필요한 layer utility 를 구현하고 `VGGNet` 클래스에 반영했다.
### filter 개수 축소
이미지가 단순한만큼, filter 의 개수도 적어도 무방하지 않을까?
### result
fc : 1024
num_epoch : 100
batch_size : 256
learning_rate : 1e-3
reg : 0

인 세팅에서 1 epoch 지났을때
train acc : 0.492
val acc : 0.507
## Sanity Check
### loss check
랜덤 X, 랜덤 y 를 설정한다. model.loss 메소드를 호출해서 값을 확인할 수 있다.
* reg=0 일때 loss가 ln(클래스_개수) 와 유사
* reg 값 조금 주면 위의 loss 보다 조금 증가

reg 값을 0.5로 줬을때 loss 가 6 수준으로 확 뛴다. parameter 들의 개수가 많아서 그런 것이다. 즉, parameter 크기가 커진만큼 내가 적절히 weight initialization 하지 않았다는 뜻이다. He 의 initialization 을 사용한다.

CONV layer 의 parameter 에서 fan_in / fan_out 을 어떻게 생각할 수 있을까? fan_in 을 **"단위 내적 연산의 dimension"** 이라고 이해한다. 그러니 CONV layer 에서는 단위 내적 연산이 filter 크기 x filter 크기 x input depth 가 될 것이다.
### gradient check