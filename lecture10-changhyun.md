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