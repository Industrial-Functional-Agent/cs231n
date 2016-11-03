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

#### Step 1: Train (or download) a classification model
For example, VGG, GoogLeNet, and ResNet can be used.
#### Step 2: Attach new fully connected "regression head" to the network
**Overfeat** and **VGG** attaches regression head after last conv layer. Otherwise, **DeepPose** and **R-CNN** attaches regression head after last fully connected layer.

Regression head is also composed of FC layers and classifier. It can be divided into **class agnostic** head and **class specific** head. In formal, regression head produces **one box** (4 numbers) for best-fit class. In latter, head produces **C boxes** (C x 4 numbers) for each classes.

If we want exactly **K boxes**, then we set regression head to produces exactly K x 4 numbers. Human pose estimation can be a good example. We can interpret each human body's joint as a rectangle (one for left arm, another for right arm ...).

Regrssion result can be in the outside of input image.

#### Step 3: Train the regression head only with SGD and L2 loss
For a given CNN, we can **fine tune** it depends on amount of training dataset for localization problem.
#### Step 4: At test time, use both heads
 
## Sliding Window
ILSVRC 우승을 위해서는, 위의 일반론적인 방법외에 특별한 것들이 더 필요했다! 하지만 내용이 쉽다! (학생들에게 프로젝트에 사용해볼 수 있을거라 안내함)
* Run classification + regression network at **multiple locations** on a highresolution image
* Convert fully-connected layers into convolutional layers for efficient computation
* Combine classifier and regressor predictions **across all scales** for final prediction

### Overfeat (2013 winner)
![overfeat sliding window (effective)](2-overfeat-effective.jpg)

In sliding window technique, **multiple scale** of input image are prepared. Therefore, each scaled image can be bigger than network input size. Then, we treat network input size as a sliding window's size, then slide over whole scaled input image, and get multiple set of results(score and box coordinates). **Overfeat** apply technique called **greedy merge boxes and scores** to set of results to get final single score and box coordinates for input image. Detail of this technique can be found in [original paper](https://arxiv.org/pdf/1312.6229v4.pdf)

Computation can be more effective by replacing FC layer to CONV layer. For big image with mulitple windows, convolution computation can be shared among shared region of windows.
(그림 그리면 좋을듯) 

### Record
|Year|Network|Descriptoion|Performance|
|:---|:-------|:-----------------------------------------------------------|-----|
|2012|AlexNet |No publication                                              |34.2%|
|2013|Overfeat|Multi scale convolutional regression with box merging       |29.9%|
|2014|VGGNet  |Fewer scales/locations, simpler method, but deepter network.|25.3%|
|2015|ResNet  |Use RPN localization method, much deeper network.           |9%   |

VGGNet back-propagate to the head, and other networks only back-propagate regression head. Some networks entirely divide classification and regression head into different networks

## Object Detection: Detection as Classification
object detection 을 regression 으로 접근하려면 난감하다. 몇개의 object 가 있을지 모르므로, output 이 가변적이기 때문이다. object 개수 x 4 numbers 를 output 으로 뽑아내야 box coordinate 를 구성할 수 있고, 그래야 score 를 뽑아낼 수 있다.

그래서 regression 대신 classification 으로 접근한다! input image 의 partial region 을 보면서 classification 을 진행한다. 기본적으로 모든 window size 에 대해서 해본다. 각각의 sub-image 에 대해서 마치 그게 input data 인 것처럼 네트워크를 통과시키면 각 class 에 대한 score 를 얻을 수 있다. (이걸 어느 threshold 값 기준으로 넘으면, 해당 object 가 detect 되는걸까?)

**problem:** Need to test many positions and scales.
**solution:** If your classifier is fast enough, just do it. 

사실 detection 은 CV 의 고전적인 문제이다. 예를 들어 pedestrian detection 의 경우 **HoG** feature 가 효과적이었다. HoG = **Histogram of Oriented Gradients** [논문 링크, 2005년꺼네?](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf)

이미지를 여러 scale 로 준비하고, 각각의 scale image 마다도 window 로 sub image 를 순회하면서 linear classifier 를 때려서 **feature pyramid** 를 획득한다. 거기에 **non-maxima suppression** 이라는 테크닉이 가미된다.

이를 기반으로 DPM(Deformable Parts Model) 을 생성한다. 알고리즘을 좋아한다면 흥미로울 거라고 하네. [논문 링크, 2010년꺼네?](https://people.eecs.berkeley.edu/~rbg/papers/Object-Detection-with-Discriminatively-Trained-Part-Based-Models--Felzenszwalb-Girshick-McAllester-Ramanan.pdf)

**Problem:** Need to test many positions and scales, and use a computationally demanding classifier (CNN)
**Solution:** Only look at a **tiny subset** of possible positions

그래서 region proposal 의 개념이 나온다.

## Region Proposals
* Find "blobby" image regions that are likely to contain objects
* "Class-agnostic" object detector
* Look for “blob-like” regions

구체적으로는 여러 기법이 가능한데, 그 중 **selective search** 가 좀 대표적인가보다.
(Bottom-up segmentation, merging regions at multiple scales)
이미지의 비슷한 segment 들을 merge 시키면서 scale 을 변화 시키면서, 매 scale 에서 이미지 -> boxes 로 변환한다. 디테일은 알 필요 없다고 한다.

그 외에도 choice 들은 많이 있다. 필요하다면, EdgeBoxes 를 생각해보라고 한다. 표를 보니, 속도/성능 모두 상위권이다.

## R-CNN
[논문 링크, 얘는 2014년이네?](https://arxiv.org/pdf/1311.2524v5.pdf)
= 모든 것을 putting it together

* input image 에 proposal method 를 적용해서 대략 2000개 정도의 RoI(Region of Interest) 를 뽑아낸다.
* warped image regions = 크기가 동일해지네? CNN 에 집어넣기 위함인듯?
* Forward each region through ConvNet
* Apply bounding-box regressions / Classify regions with SVMs

### Step 1: Train (or download) a classification model for ImageNet (AlexNet)
### Step 2: Fine-tune model for detection
- Instead of 1000 ImageNet classes, want 20 object classes + background
- Throw away final fully-connected layer, reinitialize from scratch = 마지막 FC layer 의 모든 parameter 를 weight initialization 시킨다는 얘기인듯!
- Keep training model using positive / negative regions from detection images

(여기서 positive / negtive region 이 뭔지 모르겠다~~~)
### Step 3: Extract features
- Extract region proposals for all images
- For each region: warp to CNN input size, run forward through CNN, save pool5 features to disk
- Have a big hard drive: features are ~200GB for PASCAL dataset!

### Step 4: Train one binary SVM per class to classify region features
- Step 3 에서 training image region 에 대한 pool5 feature 들을 저장해놨었다. 그 feature 들을 input 으로 해서 binary SVM 을 train 한다.

(binary SVM 이 정확히 어떤 느낌이었더라?)

### Step 5 bbox regression
- For each class, train a linear regression model to map from cached features to offsets to GT boxes to make up for “slightly wrong” proposals

offset 을 4 numbers 로 regression 한다. (0, 0, 0, 0) 이면 제일 좋고, 조금 tilt 된거면 각 원소에 값이 생긴다.

(흠... data set 의 구성과 대회 조건에 대해서 좀 알아야겠다. Step 4 에서 binary classification 하는거나 Step 5 에서 4 numbers 로 regression 할때 training data 들이 무엇인지 이해가 안된다)
(아마 사진에 여러 object 들을 각각 localization 해야할 것이다)

61 페이지까지 봤는데..... 힘들군.