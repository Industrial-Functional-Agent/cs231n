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

Regression head is also composed of FC layers and classifier. It can be divided into **class agnostic** head and **class specific** head. In formal, regression head produces **one box** (4 numbers) for best-fit class. In latter, head produces **C boxes** (`C x 4` numbers) for each classes.

If we want exactly **K boxes**, then we set regression head to produces exactly `K x 4` numbers. Human pose estimation can be a good example. We can interpret each human body's joint as a rectangle (one for left arm, another for right arm ...).

Regrssion result can be in the outside of input image.
#### Step 3: Train the regression head only with SGD and L2 loss
For a given CNN, we can **fine tune** it depends on amount of training dataset for localization problem.
#### Step 4: At test time, use both heads
## Sliding Window
* Run classification + regression network at **multiple locations** on a highresolution image
* Convert fully-connected layers into convolutional layers for efficient computation
* Combine classifier and regressor predictions **across all scales** for final prediction

### Overfeat (2013 winner)
![overfeat sliding window (effective)](2-overfeat-effective.jpg)

In sliding window technique, **multiple scale** of input image are prepared. Therefore, each scaled image can be bigger than network input size. We treat network input size as a sliding window's size, slide over whole scaled input image, and get multiple set of results(score and box coordinates). **Overfeat** apply technique called **greedy merge boxes and scores** to set of results to get final single score and box coordinates for input image. Detail of this technique can be found in [original paper](https://arxiv.org/pdf/1312.6229v4.pdf)

Computation can be more effective by replacing FC layer to CONV layer. For big image with mulitple windows, convolution computation can be shared among shared region of windows.
### Record
|Year|Network|Descriptoion|Performance|
|:---|:-------|:-----------------------------------------------------------|-----|
|2012|AlexNet |No publication                                              |34.2%|
|2013|Overfeat|Multi scale convolutional regression with box merging       |29.9%|
|2014|VGGNet  |Fewer scales/locations, simpler method, but deepter network.|25.3%|
|2015|ResNet  |Use RPN localization method, much deeper network.           |9%   |

VGGNet back-propagate to the head, and other networks only back-propagate regression head. Some networks entirely divide classification and regression head into different networks
## Object Detection: Detection as Classification
In object detection, the number of classes in input image **varies**. So if we approach object detection problem as a single regression, we can't decide exact number of outputs for network which make calculating loss function impossible. 

Therefore we approach object detection problem as **a sequence of classification**. Network's output are **scores** and **box coordinates**. We should set network input size smaller than expected input images. In training/testing, partial regions of input image which fit network input size are dealt with network.

But problem is that partial regions for multi-scaled input image are too many. Our CNN classifier is not that fast like HOG or DPM classifier. Therefore we need to focus on **"probable"** partial regions. That is, **region proposal**.
### R-CNN: CNN + Region Proposal
#### step 1: Train (or download) a classification model for ImageNet
VGGNet or ResNet are recommended.
#### step 2: Fine-tune model for detection
Instead of 1000 ImageNet classes, we only need 20 classes + 1 background. So, last fully-connected layer's output size should be changed(1000 -> 21). Then we reinitialize last fully-connected layer and **fine-tune**(train for small dataset) it.

For fine-tuning, we should prepare training data. In **PASCAL VOC** dataset, each image contains multiple set of box coordinate and label. For each image, apply region proposal(selective search for R-CNN), get multiple regions and classify each region by rule. The rule is, if there is any box coordinate has IoU larger than 0.5 with partial region, classify that region as corresponding label, or as background if not. Not only label but also offset (relative size and position with box coordinate) is also saved for step 5. Those classified partial regions become traing data for reinitialized last fully connected layer.    
#### step 3: Extract features
Extract region proposals for all images. For each region, warp to CNN inut size, run forward through CNN, and save POOL5(or layer just before FC layer) features to disk. Maybe we need a big hard drive. For example, features for **PASCAL** dataset are ~200GB.
#### step 4: Train one binary SVM per class to classify region features
#### step 5: (bbox regression) For each class, train a linear regression model to map from cached features to offsets to GT boxes to make up for "slightly wrong" proposals    

But R-CNN has some problems. First, we forwarding whole CNN for each region proposal. It is slow. Second, SVM and regressor are post-hoc. They are attached after pre-trained CNN. Third, training pipeline is too complex.
### Fast R-CNN
In Fast R-CNN, we start with CNN which has input size same to image size. Region proposal technique is applied to input image. But instead of forwarding CNN for region proposal, we only pass coordinate to get corresponding projection in the output feature map of CNN. Then we get region proposals in conv feature map which passed to fully connected layer.
#### RoI(Region of Interest) Pooling
But problem is fully connected layers expect **low resolution conv feature** and size of region proposals in conv feature map are all different. Therefore we use **pooling technique** similar with max pooling in conventional CNN. For each conv feature map of region proposal, divide it in the input size of first fully conneted layer, and max pool. Then we get proper input for first fully connected layer.

To fix second and third problems in R-CNN, we just train the whole system end-to-end all at once.
### Faster R-CNN
TODO
#### RPN
TODO
### YOLO
TODO
### Evaluation: mAP(mean average precision)
#### precision / recall
* true / false : whether guess is correct or not
* positive / negative : classifying as such a class or not 
* precision = true positive / (true positive + false positive)
* recall = true positive / (true positive + false negative)

#### mean average precision
In general, **average precision** is area under the precision-recall curve. But in ILSVRC, we only submit a single classifier. Therefore no curve exists, only a point of precision and recall. So we interpret average precision as simple product of precision and recall. 

Then mAP can be derived as average of average precision over classes