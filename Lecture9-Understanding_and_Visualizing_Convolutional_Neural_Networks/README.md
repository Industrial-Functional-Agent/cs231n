## Understanding ConvNet
### Visualize the filters/kernels
The parameters of each CONV layer composed of multiple 2D filters. So we can visualize each 2D filter as a image. But only parameters in first CONV layer are meaningful. That's because only input to first CONV layer(which is input image) has graphical meaning in 2D space.
### Visualize representation
In every CNN, their last layer is a classifier. And we call input vector to this layer **"code"**. For example, code of VGGNet has 4096 dimension.
#### t-SNE visualization
To visualize (high-dimensional) code, we embed code to two dimensional space, while pairwise distances are conserved. So similar codes end up in similar places, while dissimilar codes end up wherever.
<img src="https://lvdmaaten.github.io/tsne/examples/mnist_tsne.jpg" height="400" align="center"> 
### Occlusion Experiments
Occlusion experiments is for understanding which part of image contribute to correct label score. With given image, sliding image with small patch occlusion, and computing CNN score for true label. Then we can draw two dimensional heat map, (patch x-pos, patch y-pos) to score. Lower score mean that patch contributes classifier a lot, and higher score mean that patch disturb classifier a lot.

<img src="heat-map.jpg" height="500"/>
