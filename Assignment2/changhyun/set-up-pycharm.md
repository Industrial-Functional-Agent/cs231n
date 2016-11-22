## IPython 말고 PyCharm 에서 ConvNet 구현
open 메소드의 첫번째 파라미터로 집어넣는 디렉토리 경로는 무조건 상대경로이다. 앞에 `/` 같은걸 붙여봐도 절대경로로 인식되지 않는다. 그래서 번거로움을 피하기 위해 **main.py** 를 최상단 디렉토리에 배치했다. 

IPython 의 코드를 참고해서 간단하게 main.py 스크립트를 작성할 수 있다. data 를 만들고, model 을 생성하고, solver 를 생성하고, solver 의 train 메소드만 호출하면 된다.
## Optimization 연산 시간 측정
IPython 코드의 설정대로 model / solver 를 만들었는데, 한번 forward/backward propagate 하는 iteration 의 소요시간을 측정한다.

category|field name | value |
--------|-----------|-------|
model|weight_scale|0.001|
 |hidden_dim|500|
 |reg|0.001|
solver|batch_size|50|
 |update_rule|Adam|
 |learning_rate|1e-3|
**1.109 s** / iteration

batch_size = 50 이고 training data 크기가 49,000 였으므로 1 epoch = 980 iteration 이었다. 그러므로 대략 20분 좀 안되게 걸렸던 것.

ThreeLayerConvNet 이고, 입력하지 않았으므로

field name|value
----------|-----
input_dim|3x32x32
num_filters|32
filter_size|7
hidden_dim|100
## Configurable 한 ConvNet 클래스
`ThreeLayerConvNet` 를 구성하는 layer 목록
* CONV
* RELU
* POOL
* AFFINE
* RELU
* AFFINE
* SOFTMAX

