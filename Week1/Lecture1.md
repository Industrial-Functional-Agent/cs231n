# Lecture 1 Introduction and Historical Context
Convolutional Neural Network를 위주로 배우게 될 것이다.
인터넷 데이터의 85%가 multimedia이다. 그러나 이것은 (해석하기?) 힘든 데이터이다. 즉 인터넷의 Dark Matter이다. 지금도 YouTube에 엄청난 양의 데이터가 업로드 되고 있다. 이 엄청난 양의 데이터를 다 일일히 레이블링 하는 것은 매우 어려운 일이다. Search나 관련된 광고들을 포함시키려면, indexing혹은 레이블링을 해야 하는데, 이를 손으로 하는 것은 불가능하다. Vision technology를 통해서 이루어져야 한다.  
cs231n은 neural network와 visual recognition에 집중한다.

## Today's agenda
* A brief history of computer vision
* CS231n overview

Big Bang of Evolution 543 millon years, B.C. Animals start to diversify. It was onsef of the eyes.  

Camera Obscura is first camera(?) engineering of vision.  

How does visual work in our biological brain?
Visual cortex에 electrode를 꽂아서 전기 신호를 측정했다. Primary Visual Cortex는 eye에서 제일 멀리 떨어져 있다. 물고기나 쥐를 보여줬을 때는 neuron이 spike하지 않았지만, 오히려 그 slide를 교체하기 위해 slide를 빼거나 집어 넣었을 때에 neuron이 spike하였다. Slide의 edge가 움직 일때 neuron은 spike한다. Visual cortex의 column(?)들은 oriented bars에 stimulated되지, fish나 mouse에 stimulated되지 않았다. Beginning of visual processing은 fish나 mouse전체를 인식하는게 아니라 simple structure를 인식하는 것이다. Deep neural network의 modeling도 이렇게 할 것이다?  

Block world
물체의 edge를 뽑아내려 하였다. 첫번째 computer vision PhD 논문  

1966년이 MIT AI Lab에서 Vision을 Summer project로 끝내려 했다. Birth Day of Computer Vision.

Vision의 저자인 David Marr, Build hierarchical model of vision processing? Several layer가 존재 한다. Simple Structure부터 시작해서 complex structure가 된다. 2d로 먼저하고 3d  

Perceptual grouping. 아직도 해결 못했다. 

Stanford work 는 3d shape of object를 modeling하려 했다. 지금은 object가 무엇인지 recognizing하려한다. Focusing the recogntion. 
Computer vision이 다시 AI가 중요해진 이유.

물체를 인식하는데 있어서 전체 모습이 필요한가? Feature를 뽑아서 그것을 인식하면 되는 것 아닌가? decision making이 매우 빨라야 하니까. 그래서 Deep Learning이전에는 이런 feature를 찾는것이 목적이었다.(?) 그러나 Deep Learning은 그 feature를 알아서 찾는다.

Benchmark가 필요하다!(Standard image set이 필요하다?) IMAGENET
2012년 Imagenet에서 CNN이 다 강간했다. 그러나 그 아이디어는 70~80년대에 나왔던 것이다. This is beginning of deep learning

cs231n foucses on one of the most important problems of visual recognition - image classification

여러 문제들에 대한 flavor들을 알려줄 것이다.
cnn is not invented overnight
capcity of network?

많은 데이터는 그자체로 overfitting을 피하게 해준다.

image classification이 다가 아니다. 나아갈게 많다. ㅎㅎ... 