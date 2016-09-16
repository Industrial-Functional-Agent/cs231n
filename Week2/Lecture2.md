# Lecture 2: Image Classification pipeline
숙제나왔습니다~  
Image Classification: a core task in Computer Vision.
Image classification을 할 줄 알면, 그 위에 조금만 쌓으면, object detection, segmentation 등등 여러가지를 할 수 있다.

The problem: Semantic gap
이미지는 3D arrays of numbers, with integers between[0, 255].

E.g.
300 * 100 * 3 (3 for 3 color channels RGB)

이러한 숫자가지고 이 이미지가 cat이라는 것을 classify해야 하므로 매우 어렵다. 또한 카메라의 각도, 고양이의 각도, brightness등등이 다 달라지면 array안의 값이 다 달라진다. 어떤 상황에서도 classify할 수 있게 robust해야한다.

Challenges: Illumination
Challenges: Deformation
Challenges: Occlusion -> 전체 모습을 안보고도 cat임을 알아내야 한다.
 ...
 여러가지 challenge들이 존재 
 
 An image classifier
 정확한 algorithm이 존재하는게 아니다.
 data driven approach를 해야한다.
 
 k nearest neighbor를 뽑아서 ranking을 한후 제일 좋은 것으로 classify?
 
 우선 L1 distance(혹은 Manhatton distance)를 사용한다. 즉 test image가 들어오면 training image들과 하나하나 다 L1 distance값을 구한다. 그래서 그 값들중 가장 작은 k개만 유지한다. 
 
Nearest Neighbor classifer: lineary slower
train time: 매우 빠름
test time: 매우 느림

Neural Network는 반대이다.

The choice of distance is a **hyperparameter** common choices:

knn하면  boundary가 smooth된다.