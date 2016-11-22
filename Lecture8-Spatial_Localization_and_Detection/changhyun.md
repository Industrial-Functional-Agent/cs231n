* Computer Vision Tasks
* Localization as Regression
	* Sliding Window
* Detection as Classification
	* Region Proposal
	* R-CNN
	* Fast R-CNN
	* Faster R-CNN
	* YOLO

## Computer Vision Tasks
#### Classification
We have stuided so far. 
#### Classification + Localization
Again, this problem is for **single object**. Find and localize a object in input image, and classify it in given classes label. Localization can be represented as **4 numbers** composing rectangular box.
#### Object Detection
Object detections is a **multiple objects version** of classification + localization problem. Even though problem looks similar, detail of implementations are much different.
#### Instance Segmentation
It's not covered in this lecture.

## Localization as Regression
* **Output:** box coordinate (4 numbers) define rectangle.
* **Evaluation metric:** intersection over union.

### General Steps
#### step 1: Train (or download) a classification model
For example, VGG, GoogLeNet, and ResNet can be used.
#### step 2: Attach new fully connected "regression head" to the network
**Overfeat** and **VGG** attaches regression head after last conv layer. Otherwise, **DeepPose** and **R-CNN** attaches regression head after last fully connected layer.

Regression head is also composed of FC layers and classifier. It can be divided into **class agnostic** head and **class specific** head. In formal, regression head produces **one box** (4 numbers) for best-fit class. In latter, head produces **C boxes** (C x 4 numbers) for each classes.

If we want exactly **K boxes**, then we set regression head to produces exactly K x 4 numbers. Human pose estimation can be a good example. We can interpret each human body's joint as a rectangle (one for left arm, another for right arm ...).

Regrssion result can be in the outside of input image.
#### step 3: Train the regression head only with SGD and L2 loss
For a given CNN, we can **fine tune** it depends on amount of training dataset for localization problem.
#### step 4: At test time, use both heads
### Sliding Window
ILSVRC 우승을 위해서는, 위의 일반론적인 방법외에 특별한 것들이 더 필요했다! 하지만 내용이 쉽다! (학생들에게 프로젝트에 사용해볼 수 있을거라 안내함)
* Run classification + regression network at **multiple locations** on a highresolution image
* Convert fully-connected layers into convolutional layers for efficient computation
* Combine classifier and regressor predictions **across all scales** for final prediction

#### Overfeat (2013 winner)
![overfeat sliding window (effective)](2-overfeat-effective.jpg)

In sliding window technique, **multiple scale** of input image are prepared. Therefore, each scaled image can be bigger than network input size. Then, we treat network input size as a sliding window's size, then slide over whole scaled input image, and get multiple set of results(score and box coordinates). **Overfeat** apply technique called **greedy merge boxes and scores** to set of results to get final single score and box coordinates for input image. Detail of this technique can be found in [original paper](https://arxiv.org/pdf/1312.6229v4.pdf)

Computation can be more effective by replacing FC layer to CONV layer. For big image with mulitple windows, convolution computation can be shared among shared region of windows.
(그림 그리면 좋을듯) 

#### Record
|Year|Network|Descriptoion|Performance|
|:---|:-------|:-----------------------------------------------------------|-----|
|2012|AlexNet |No publication                                              |34.2%|
|2013|Overfeat|Multi scale convolutional regression with box merging       |29.9%|
|2014|VGGNet  |Fewer scales/locations, simpler method, but deepter network.|25.3%|
|2015|ResNet  |Use RPN localization method, much deeper network.           |9%   |

VGGNet **back-propagate to the head**, and other networks only **back-propagate regression head**. Some networks entirely divide classification and regression head into different networks

## Detection as Classification
object detection 을 regression 으로 접근하려면 난감하다. 몇개의 object 가 있을지 모르므로, output 이 가변적이기 때문이다. object 개수 x 4 numbers 를 output 으로 뽑아내야 box coordinate 를 구성할 수 있고, 그래야 score 를 뽑아낼 수 있다.

그래서 regression 대신 classification 으로 접근한다! input image 의 partial region 을 보면서 classification 을 진행한다. 기본적으로 모든 window size 에 대해서 해본다. 각각의 sub-image 에 대해서 마치 그게 input data 인 것처럼 네트워크를 통과시키면 각 class 에 대한 score 를 얻을 수 있다. (이걸 어느 threshold 값 기준으로 넘으면, 해당 object 가 detect 되는걸까?)

**problem:** Need to test many positions and scales.
**solution:** If your classifier is fast enough, just do it. 

사실 detection 은 CV 의 고전적인 문제이다. 예를 들어 pedestrian detection 의 경우 **HOG** feature 가 효과적이었다. HOG = **Histogram of Oriented Gradients** [논문 링크, 2005년꺼네?](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf)

이미지를 여러 scale 로 준비하고, 각각의 scale image 마다도 window 로 sub image 를 순회하면서 linear classifier 를 때려서 **feature pyramid** 를 획득한다. 거기에 **non-maxima suppression** 이라는 테크닉이 가미된다.

이를 기반으로 **DPM(Deformable Parts Model)** 을 생성한다. 알고리즘을 좋아한다면 흥미로울 거라고 하네. [논문 링크, 2010년꺼네?](https://people.eecs.berkeley.edu/~rbg/papers/Object-Detection-with-Discriminatively-Trained-Part-Based-Models--Felzenszwalb-Girshick-McAllester-Ramanan.pdf)

**Problem:** Need to test many positions and scales, and use a computationally demanding classifier (CNN)
**Solution:** Only look at a **tiny subset** of possible positions

그래서 region proposal 의 개념이 나온다.

### Region Proposals
* Find "blobby" image regions that are likely to contain objects
* "Class-agnostic" object detector
* Look for “blob-like” regions

구체적으로는 여러 기법이 가능한데, 그 중 **selective search** 가 좀 대표적인가보다.
(Bottom-up segmentation, merging regions at multiple scales)
이미지의 비슷한 segment 들을 merge 시키면서 scale 을 변화 시키면서, 매 scale 에서 이미지 -> boxes 로 변환한다. 디테일은 알 필요 없다고 한다.

그 외에도 choice 들은 많이 있다. 필요하다면, **EdgeBoxes** 를 생각해보라고 한다. 표를 보니, 속도/성능 모두 상위권이다.

### R-CNN
[논문 링크, 얘는 2014년이네?](https://arxiv.org/pdf/1311.2524v5.pdf)
= 모든 것을 putting it together

* input image 에 proposal method 를 적용해서 대략 2000개 정도의 RoI(Region of Interest) 를 뽑아낸다.
* warped image regions = 크기가 동일해지네? CNN 에 집어넣기 위함인듯?
* Forward each region through ConvNet
* Apply bounding-box regressions / Classify regions with SVMs

그러므로 ConvNet 에는 classifier 랑, 4 numbers 만 뽑아내는 regression head 가 있으면 된다.
#### step 1: Train (or download) a classification model for ImageNet (AlexNet)
#### step 2: Fine-tune model for detection
- Instead of 1000 ImageNet classes, want 20 object classes + background. 즉 마지막 FC layer 가 4096 x 1000 에서 4096 x 21 이 된다. 21 = 20 categories + 1 background 이다. FC layer 가 완전히 바뀌므로 parameter 도 던지고 새롭게 reinitialize 해서 scratch 에서부터 다시 learning 한다.
- Keep training model using positive / negative regions from detection images

(여기서 positive / negtive region 이 뭔지 모르겠다~~~)
#### step 3: Extract features
- Extract region proposals for all images
- For each region: warp to CNN input size, run forward through CNN, save pool5 features to disk
- Have a big hard drive: features are ~200GB for PASCAL dataset!

#### step 4: Train one binary SVM per class to classify region features
- step3 에서 training image region 에 대한 pool5 feature 들을 저장해놨었다. 그 feature 들을 input 으로 해서 binary SVM 을 train 한다.

(binary SVM 이 정확히 어떤 느낌이었더라?)

#### step 5 bbox regression
- For each class, train a linear regression model to map from cached features to offsets to GT boxes to make up for “slightly wrong” proposals

offset 을 4 numbers 로 regression 한다. (0, 0, 0, 0) 이면 제일 좋고, 조금 tilt 된거면 각 원소에 값이 생긴다.

(흠... data set 의 구성과 대회 조건에 대해서 좀 알아야겠다. step 4 에서 binary classification 하는거나 step 5 에서 4 numbers 로 regression 할때 training data 들이 무엇인지 이해가 안된다)
(아마 사진에 여러 object 들을 각각 localization 해야할 것이다)

#### Evaluation
mAP(mean average precision) 이라는 척도를 사용한다.
R-CNN 이 큰 jump 였고, 거기에 bbox 를 얹으니 조금 향상됬고, VGGNet 이 도입되니 더 큰 향상이 있었다.
#### Problem
* R-CNN 은 느리다. 각각의 region proposal 에 대해서 ConvNet 의 full forward pass 를 돌려야 한다.
* 참고로 SVMs 랑 regressor 는 post-hoc 이라서 ConvNet parameter 의 update 에는 영향을 미치지 않는다.
* complex multi-stage training pipeline

그래서 Fast R-CNN 이 도입된다. 
### Fast R-CNN (ICCV 2015)
순서를 바꾼다. input image 에서 RoI 를 추출하지 않고, ConvNet 을 통과시켜서 얻은 feature map (특히 CONV5 map) 에서 RoI 를 추출한다.

#### RoI Pooling
hi-res input image 에서는 hi-res conv feature 가 나온다. 크기는 `C x H x W`. 그러나 FC layer 에서는 `C x h x w` 를 입력으로 기대하고 있어서 크기가 안맞는다. 이 경우에 대해서 pooling 으로 문제를 해결한다. `H x W` 를 `h x w` 개수의 grid 로 나누고, grid 안에서 max pooling 하는 것. back propagation 에서도 POOL layer 에 대해 했던것과 동일하게 하면 된다.
#### 장점 / 단점
장점은 R-CNN 에 비해서 매우 빠르다는 점이다. 테스트 타임이 유독 더 빠른데 왜 그렇게 됬는지 이해를 못했다.

단점은 사실 테스트 타임에 Region Proposal 하는 시간은 고려되지 않았다는 점이다. 이를 고려하면 이미지 당 0.32s 이던게 2s 가 되버려서 real time 이 힘들어진다. **selective search** 기준이다.

이에 대한 해결책은 Region Proposal 마저도 ConvNet 이 하도록 만드는 것이다(?)
### Faster R-CNN
ConvNet 을 돌려서 만들어진 feature map 에다가 **RPN(Region Proposal Network)** 가 붙는다. RPN 을 통과한 후에는 Fast R-CNN 과 마찬가지로 아래 세 과정을 거친다.
* RoI Pooling
* upstream classifier
* bbox regressor

#### Region Proposal Network
ConvNet 을 거친 feature map 이 있다. 작은 window 크기(`3x3`)로 sliding window 를 한다. 그래서 256-d 벡터가 나오는데 잘 이해가 안된다. 3x3 크기의 window 에 3x3 크기의 filter 256개 때려서 `1x1x256` 크기의 CONV layer 가 나오는가 싶다. 아무튼 여기서 두 갈래로 나눠져서 object/non-object 를 분류하는 score 를 매기는 classifier 와, box location 에 대한 regressor 가 training 된다(?)

each position 에 translation invariant 한 anchor box 를 N개 쓴다는데 무슨말인지 모르겠다.

original paper 에서는 RPN training ---> ConvNet training ---> magical merge 였는데 좀 messy 했다고 한다. 그래서 published 이후에는 함께 training 되게 바뀌었다. 그러니 총 4개의 loss 를 objective 로 training 된다. 
* RPN classification (anchor good/bad)
* RPN regression (anchor -> proposal)
* Fast R-CNN classification (over classes)
* Fast R-CNN regression (proposal -> box)

#### Result
Faster R-CNN 으로는 이미지 당 테스트 타임이 0.2s 까지 가능했다. 
#### Good Question
RoI Pooling 에서 회전할 경우(?) 문제가 발생한다. 2015 여름에 DeepMind 에서 나온 매우 HOT 한 논문에 따르면, RoI 대신 bi-linear interpolation 을 써서 이 문제를 해결한다(?). 하지만 아직 whole pipeline 에 정식으로 포함되지는 않았다.
#### State of the Art
ResNet 101 (layer 개수) + Faster R-CNN + some extras 가 킹왕짱이다.
some extras 에는,
* box refinement
* context : whole image 의 feature 를 줌(?)
* multi-scale testing : different size 의 image

### YOLO
진짜 빠른데, 성능은 Faster R-CNN 에 비해서 별로다. 그래도 빠르니 프로젝트에 추천한다고 했다.

PPT 마지막에 정리가 있구나.