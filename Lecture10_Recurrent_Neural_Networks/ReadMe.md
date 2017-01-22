|architecture          |example                            |description|
|----------------------|-----------------------------------|-----------|
|one-to-many           |image captioning                   |image -> seq of words|
|many-to-one           |sentiment classification           |seq of words -> sentiment|
|many-to-many(shifted) |machine translation                |seq of words -> seq of words|
|many-to-many(parallel)|video classification on frame level|prediction of previous frame help predict current frame|

To translate sentence to sentence in another language, **"context"** might be needed. It means that some amount of words have to be prepared to start translation. That's why machine translation need **shifted** architecture.

DRAW : A Recurrent Neural Network For Image Generation

## Vanilla RNN
**new\_hidden\_state** = *map\_hh*(**old\_hidden\_state**, **input**)  
**output** = *map\_yh*(**new\_hidden\_state**)

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

* `W_xh` has dimension (V, H)
* `W_hh` has dimension (H, H)
* `W_hy` has dimension (H, V)

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

### [TODO] Questions
Several things didn't get understood yet. Discussion and studying introduced paper are needed.

* How RNN compute lookup key in each time step?
* How exactly "weighted sum" of activation map is computed?
* It seems like output of CNN affects every time step of RNN, which differ with basic compound model that output of CNN affects only first hidden state of RNN in basic compound model. Then how exactly output of CNN affects components of RNN?

## More Complex RNN
### Stacked RNN
The recursion formula of stacked RNN is exactly same with vanilla RNN's one, only representations are different. The parameter matrix `W_hx` and `W_hh` are stacked horizontaly (**n x 2n**), and two input vectors `h(l-1, t)`, `h(l, t-1)` for stacked RNN, are stacked vertically(**2n x 1**). Each network in specific depth has its own parameters, but same among time steps as original RNN is.

In fact, we don't use vanilla RNN or stacked vanilla RNN in practice any more. As neural net goes deeper and more complicative, we need more advanced RNN architecture. That's why lecture lead us to **LSTM**. 

### LSTM (Long Short Term Memory)

* why we multiply `input` to `g` even though `g` already exists between -1 and 1?
    * We multiply `input` to `g` to make adding term have richer representation. `g` is tanh of linear transformation, and this single non-linearity may not be enough. Those two terms could be understand as `g` for amount and `input` for strength of variation to the cell state.
* Be aware of that hidden state in specific time step is used both in computing next `f`, `i`, `g`, `o` and computing higher hidden state or prediction.
* Vanilla RNN transform from hidden state to hidden state while LSTM evolve in additive manner and also forget some amount, which reminds us **ResNet**.
    * Additive interaction works as super-fast **highway** in back-propagation.
* Vanilla RNN has vanishing gradient problem.

### Repeating multiplication of W_hh

In back-propagation of vanilla RNN, the parameter `W_hh` multiplies once in every iteration. The involution of same number usually concludes in two cases, **explosion** or **vanishment**. The former one can be controlled by **gradient clipping** (which does not mentioned in lecture) technique, while the latter one can be controlled by **LSTM**. LSTM tends to vanish slowly than vanilla RNN thanks to super-fast highway in their recursion formula.  

### What about ReLU?

In CNN, our basic choice of non-linearity is ReLU. But it seems to be hard to adapt ReLU non-linearity in LSTM recursion formula. `g` might be replaced by ReLU, but then adding term of hidden state would be computed in single direction.

### Variant and Friend of LSTM

As LSTM show better performance than vanilla RNN, many researchers have tried to improve recursion formula of LSTM, therefore many variants was created. In lecture, Andrej recommend **GRU** for its **shorter formula**, just **single type of hidden state** and **compatible performance** with LSTM.