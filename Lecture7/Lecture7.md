## Convolution Layer
The convolution layer is a basic building block of CNN. Convolution layer has small filters that are parameters. It takes input image, convolves that image with each filters, stack them, and finally makes new output image, which called **activation map**.

### Example
Let's deal with **CIFAR-10** image as example, which we used in assignment 1. Each CIFAR-10 image has **32 x 32** width and height, and **3** channels in depth for representing RGB. Totally it is a **32 x 32 x 3** tensor. And assume that we have **5 filters** with **5 x 5** size.
The depth of filter usually equals to input image's depth. So we have **5 x 5 x 3** dimension filter in this case, then we can take **dot product** between partial image and filter. Additional **bias** is taken care to compute each pixel in activation map. Finally we will get **28 x 28 x 1** activation map for each filter. Repeat this process for each filter, stack all the activation map, and we finally get output image. 

![convolution layer](1.jpg)

## Convolutional Neural Network
**Convolutional neural network** composed of many **convolution layers**, interbed with **activation function** (e.g ReLU) layer. Additional layer called **pooling layer** will be discussed. As input data pass through layers, it transformed from **low-level** feature to **high-level** feature. In image classification problem, after extracting high-level feature from input image, **trainable classifier** is attached to final layer. That's it.

### Hyperparameter
Almost conventional framework (e.g. Torch, Caffe, Lasagne) shares same hyperparameter configuration for CNN. There are totally four hyperparameters to define exact convolution computation.

**# of filters.** A number of filters equals to output tensor's depth. Therefore, it will be same with next convolution layer's filter depth.
**Filter size.** Usually, filter is a square and has odd size. **3 x 3** or **5 x 5** is usual. A combination of (filter size, stride) should match input image's size.
**Stride.** In former example, we assumed that filter convolves entire input image pixel by pixel. This equals to assuming stride is one. In general, stride means step size for convolution of a filter. In usual, it equals to `1`.
**Zero-padding.** If we don't use zero-padding, size of image decreases (F-1) for each layer. Nowadays number of layers in deep neural net is ~ 100 order, so (F-1) decrease in each layer should be a problem. So usually we adapt `(F-1)/2` zero-padding to maintain size of image.

### Number of parameters in each layer
It depends on input's `depth`, `filter size`, and `# of filters`. For example, considering input image with RGB channel and ten 5 x 5 filters.
> \# of parameters = (filter size x depth + bias) x # of filters = (5 x 5 x 3 + 1) x 10 = 760

### Relation between input/output size
It depends on `filter size`(=F), `stride`(=S), `# of filters`(=K). If a size of input is `W1` x `H1` x `D1`, then size of output will be,
* `W2` = (`W1` - `F` + `2P`) / `S` + 1
* `H2` = (`H1` - `F` + `2P`) / `S` + 1
* `D2` = `K`