# Gallery

This folder stores visualization of volume element of all experiments with different seeds. This includes:

- Synthetic Experiments:
  - [xor](xor): a netowrk (architecture $[2, w, 2]$ where $w$ is the number of hidden units) classifying an XOR data;
  - [sindata](sindata): a network learns a sinusoidal boundary $y = \frac{3}{5} \sin(7x - 1)$ with architecture $[2, w, 2]$; also include experiments with multiple hidden layers with architecture $[2, 8, 8, 8, 2]$ and we visualize the volume element after each of the hiddn layer;

- Real Dataset
  - [mnist_sigmoid](mnist_sigmoid): a network with $[784, 2000, 10]$ with sigmoid activation to classify MNIST digits;
  - [mnist_relu](mnist_relu): same architecture but with ReLU activation;
  - [cifar_gelu](cifar_gelu): ResNet34 on CIFAR-10 images with GELU activation;
  - [cifar_relu](cifar_relu): ResNet34 on CIFAR-10 images with ReLU activation;
  - [cifar_repeat](cifar_repeat): experiment to show that our interpolation does not respect the data geoemtry (interpolate among the same type of images, and the center is no longer of that type).
