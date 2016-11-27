## Data Augmentation
성공적인 ConvNet 들은 모두 이 스킬을 사용했다.
### horizontal filps
numpy 에서도 쉬은 테크닉이다.
### random crops/scales
ConvNet 의 input size 가 동일하므로, crop 후에 본래 크기로 scale 하는 과정이 이어진다.

그러나 train 과정에서 crops/scales 테크닉을 사용할 경우, test 과정에서는 test image 의 전체 image 를 사용하는것은 unfair 하게 느껴진다. 그러므로 test 과정에서도 average a fixed set of crops 한다. 

예로 ResNet 은 crop 뿐 아니라 애초에 input image 의 multiple-scale 을 준비하고 각각에 대해서 crop 해서 사용했다.
### color jitter
randomly jitter contrast 하는게 일반적인 테크닉이다. 

좀 더 complex 한 방법인 "color offset" 을 뽑아내는 방법이다. 각 pixel 은 [R, G, B] 로 3차원 벡터이다. 그 벡터들을 3차원 공간상에 뿌리고 PCA 해서 principal component direction 을 획득하고 그 축 상에서 sampling 한다. sampling 된 3차원 벡터를 각 pixel 에 더해서 jitter 하는 것이다. AlexNet 에 처음 소개됬고, ResNet 에도 사용됬다.
### get creative!
random mix/combinations of
- translation
- rotation
- stretching
- shearing
- lens distortions