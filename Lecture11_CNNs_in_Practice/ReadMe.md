# Data Augmentation

Data augmentation is just adding little step to transform image in some way before passing it to the CNN.

- Change the pixels without changing the label
- Train on trasformed data
- VERY widely used

Data augmentation lets you artificially expand your training set. This expanded training set is correlated but it helps training the bigger models with less overfitting. Example:

1. Horizontal flips - is really easy and very widely used.
2. Random crops/scales
	* Training: sample random crops/scales
	* ResNet: 
		1. Pick random L in range [256, 480]
		2. Resize training image, short side = L
		3. Sample random 224 x 224 patch
 	* Testing: average a fixed set of crops  
 	* ResNet: 
 		1. Resize image at 5 scales: {224, 256, 384, 480, 640}
 		2. For each size, use 10 224 x 224 crops: 4 corners + center, + flips
 		3. Then average the scores
 
3. Color jitter 
	* Simple: Randomly jitter contrast
	* Complex:
		1. Apply PCA to all [R, G, B] pixels in training set
		2. Sample a "color offset" along principal component directions
		3. Add offset to all pixels of a training image

## A general theme:
For preventing overfitting as a regularizer, during training network, we add some kind of weird stochastic noise to mess with the network. At test time we averages out this noise. 

1. Training: Add random noise
2. Testing: Marginalize over the noise 

## Data Augmentation: Takeaway
It's usually really simple to implement so you should almost always use it. It's very useful especially for small datasets. And it also fits into framework of noise / marginalization

# Transfer Learning
If the datasets are very small, use cnn as a feature extractor and re-train only the last layer. However, if the datasets are large enough, do fine tuning the final layers of cnn. The number of layers which is fine tuned is depending of how large datasets you have. Since freezed layers are very low level feature extractors (such as line), transfer learning is possible.


Tip: use only ~1/10th of the original learning rate in finetuning top layer, and ~1/100th on intermediate layers. Because the last layer is randomly initialized, it has a very large gradient and can destroy the intermediate layer. So freeze the intermediate layer, then converge the last layer, and then fine tuning the intermediate layer.

|      | very similar dataset | ver different dataset |
|:----:|:----:|:------:|
| very little data| Use Liner Classifier on top layer | You're in troubel ... Try linear classifier from different stages.|
| quite a lot of data | Finetune a few layers | Finetune a larger number of layers|

## Transfer Learning: Takeaway
Have some dataset of interset but it has < ~1M images?

1. Find a very large dataset that has similar data, train a big ConvNet there.
2. Transfer learn to your dataset


# All About Convolutions
## Part1: How to stack them
