|architecture          |example                            |description|
|----------------------|-----------------------------------|-----------|
|one-to-many           |image captioning                   |image -> seq of words|
|many-to-one           |sentiment classification           |seq of words -> sentiment|
|many-to-many(shifted) |machine translation                |seq of words -> seq of words|
|many-to-many(parallel)|video classification on frame level|prediction of previous frame help predict current frame|

To translate sentence to sentence in another language, **"context"** might be needed. It means that some amount of words have to be prepared to start translation. That's why machine translation need **shifted** architecture.

DRAW : A Recurrent Neural Network For Image Generation

## Vanilla RNN
**New State** = *func1_with_params*(**Old State**, **Input**)
**Output** = *func2_with_params*(**New State**)

## Character-level language model
In character-level language model, **training sequence**(ex: "hello") is given for training parameters in *recursion formulas*. In input layer, **each distinct character** in training sequence is mapped to *one-hot encoded* vector. The dimension of vector is equal to size of **vocabulary**, a set of distinct characters.

For example,
* 'h' is mapped to [1, 0, 0, 0]
* 'e' is mapped to [0, 1, 0, 0]
* 'l' is mapped to [0, 0, 1, 0]
* 'o' is mapped to [0, 0, 0, 1] 

### sequence length
The hyperparameter *sequence length* mean how long we will unfold RNN to compute loss function and forward/backward propagation. Because training sequence can be very long, it might happen that keeping all hidden states and outputs for entire sequence is impossible. That's why we need to decide sequence length, and train RNN consecutively for divided chunks of traning sequence.

### dimension of weight matrices
Another hyperparameter, hidden size(H) and size of vocabulary(V) determine the dimension of weight matrices.

* W_xh has dimension (V, H)
* W_hh has dimension (H, H)
* W_hy has dimension (H, V)

### sampling
We can test RNN model in any time during training process by sampling character sequence using model at that time. This sampling procedure takes **starting hidden state**, **parameters**, and **seed character** as input. Output of RNN model represents **log probabilities** of each character in vocabulary according to input character and hidden state. So we can recurrently sample characters by using sampling result of this this step as a input of next step. 

### Q&A
* hidden states are initiated by zeros.
* NO regularization adapted in *min-char-rnn.py*

### examples
* Shakespeare
* algebra
* linux code
    * syntax of generated code is almost perfect.

## Image Captioning
We can cascade CNN and RNN to caption images. In lecture, Karpathy explain with CNN especially VGGNet.

* We don't use last FC layer and softmax layer of CNN. Instead, we use output of **second from behind FC Layer**, which has 4096 dimension in the case of VGGNet.
* **Recursion formula** for new hidden state is changed. Output from CNN also contributes to updated hidden state, so additional parameters(`W_ih`) are used.
* RNN's input sequence starts with special **starting vector**. It indicates RNN that this is starting of sequence. After that, input sequence are **sampled** from previous step's output probability distribution.
* RNN's output has size equals to size of vocabulary **plus one**, for **\<END\>** token which indicates termination of recursion. 
* CNN + RNN compound is treated as a single model. That means CNN's parameters also be trained in backward pass.

For a given input image, output of FC layer in CNN(w/o last FC layer & softmax layer) might contain some information of special features in image(e.g. man's hat with some texture). It affects initial hidden state, and that hidden state affects output vector. Then next input vector might be sampled using output vector of previous step as probability distribution. In this point, we have to decide in which method be used in mapping sampled vocabulary index to input vector. The simplest method could be **one-hot encoding**.

### Microsoft COCO
This is a first (and also might be last?) dataset for image captioning introduced in class. It was created using Amazon Mechanical Turk service. 

* ~ 120K images
* ~ 5 sentences each

### Soft Attention
Beside basic compound model adding CNN and RNN, there can be fancier architecture which might achieve better performance. One of them is previewed in lecture, called **soft attention**.

In soft attention model, RNN generate not only word sequence, but also another vector which play role as **lookup key**, which has equal dimension with depth of output activation map of CNN. This lookup key computes **attention** for each pixels in activation map(e.g. dot product lookup key and each pillar of activation map, then compute softmax), which is used in computing weighted sum of activation map over pixels.

### [TODO] Tasks
Several things didn't get understood yet. Discussion and studying introduced paper are needed.
* How RNN compute lookup key in each time step?
* How exactly "weighted sum" of activation map is computed?
* It seems like output of CNN affects every time step of RNN, which differ with basic compound model that output of CNN affects only first hidden state of RNN in basic compound model. Then how exactly output of CNN affects components of RNN?

