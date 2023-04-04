![Image](vippdiism.png)

# JMA: A General Close-to-Optimal Targeted Adversarial Attack with Improved Efficiency
Adversarial examples represent a significant threat for deep learning systems. However, most of the adversarial examples proposed so far are highly suboptimal and typically rely on increasing the likelihood of the target class (or reducing the likelihood of the true class in the untargeted case), thus implicitly focusing the one-hot encoding scenario. In this paper, a more general, close to optimal, attack is proposed that resorts to the minimization of a Jacobian-induced MAhalanobis distance (JMA) term, with the Jacobian matrix taking into account the effort (in the input space) in moving along a given direction of the feature space. The experiments we carried out confirm that the proposed attack is very general and can work under a wide variety of output schemes, including the encoding-based schemes (ECOC) and multi-label classification. Moreover, the adversarial attack is often performed in very few iterations, thus proving the improved efficiency compared to existing adversarial attacks, also in the one-hot encoding case.

This is the implementation of the paper #todo{update later}:
~~~
@misc{https://doi.org/10.48550/arxiv.2208.10973,
  doi = {10.48550/ARXIV.2208.10973},
  url = {https://arxiv.org/abs/2208.10973},
  author = {Tondi, Benedetta and Costanzo, Andrea and Barni, Mauro},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Cryptography and Security (cs.CR), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Robust DNN Watermarking via Fixed Embedding Weights with Optimized Distribution},
  publisher = {arXiv},
  year = {2022},
}
~~~
Download PDF from [ArXiv](https://arxiv.org/abs/2208.10973). #todo{update later}

## Installation

Use the provided *environment.yml* to build the conda environment, then activate it:
~~~
# for win user
conda env create -f environment_win.yml
# for linux user
conda env create -f environment_linux.yml
# activate env
conda activate bowen_en
~~~

## Datasets
Our experiemnts involve following four datasets:

1. CIFAR10: it is loaded via the keras function `keras.datasets.cifar10.load_data()'.

2. GTSRB: it is be downloaded from the [official website](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip).

3. MNIST: it is loaded via the keras function `keras.datasets.mnist.load_data()'.

4. VOC2012: it is downloaded from the [offical website](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)

## JMA attack on different cases
JMA attack is implemented in three cases: ECOC, Multi-label and one-hot scenarios.
### Case 1: ECOC
In ECOC case, we implement our JMA attack in three different tasks: CIFAR10, GTSRB, and MNIST.
 
For each task, the output of the corresponding DNN model is encoded by Hadamard code. Specifically, 16-bit hadamard code is utilised to encode the output of CIFAR10 and MNIST task, while 32-bit hadamard code is used for GTSRB task. 

#### CIFAR10 task
In the CIFAR10 task, we randomly choose 200 samples from the test dataset.
 
The relevant indics are stored in 'Dataset/cifar10/to_be_attacked_200_cifar10_testset_June26_index.npy'.`

For each sample, the target class label is stored in 'Dataset/cifar10/to_be_attacked_200_cifar10_testset_June26_error_pattern.npy'.

The trained model with 16-bit hadamard encoded output is stored in 'Model/cifar10/model_weight'.

The demo code can be run as follows, where -s and -i are the parameters of step size and max iteration.
```
python -m ECOC.cifar10.JMA_CIFAR10 -s 0.5 -i 100
```

#### GTSRB task
In the GTSRB task, we only find out 32 classes over 64, which have more samples than the left. The labels of 32 classes are stored in 'Dataset/GTSRB/gtsrb_top32category_label.npy'.

Like CIFAR10 task, we also randomly choose 200 samples from the test dataset, whose indices are stored in 'Dataset/GTSRB/to_be_attacked_200_gtsrb_testset_index_June16.npy'.

The corresponding target labels are stored in 'Dataset/GTSRB/to_be_attacked_200_gtsrb_testset_error_patterns_June16.npy'.

The trained model with 32-bit hadamard encoded output is stored in 'Model/gtsrb/model_weight'.

The demo code can be run as follows, where -s and -i are the parameters of step size and max iteration.
```
python -m ECOC.gtsrb.JMA_GTSRB -s 0.5 -i 100
```

#### MNIST task
In the MNIST task, we random choose 200 samples from the test dataset, and the corresponding indices are stored in 'Dataset/mnist/3rd_tobe_attacked_mnist_testset_index.npy'.

The corresponding target labels are stored in 'Dataset/mnist/3rd_tobe_attacked_mnist_testset_error_pattern.npy'.

The trained model with 16-bit hadamard encoded output is stored in 'Model/mnist/model_weight'.

The demo code can be run as follows, where -s and -i are the parameters of step size and max iteration.
```
python -m ECOC.mnist.JMA_MNIST -s 0.5 -i 100
```


### Case 2: Multi-label
In multi-label case, we randomly choose 200 to-bt-attached sampel from the test dataset.

The chosen indices are stored in 'Dataset/voc2012/x_idfs.npy'.

There are four types of target labels: real error, 5-bit, 10-bit and 20-bit random flip error. 

The trained model weight is stored in 'Model/voc2012/model4voc2012.h5'.

The demo with real error as target labels can be run as follows, where the relevant parameters are max iteration 200 and step size 0.5
```
python -m MultiLabel.JMA_VOC -l Real -i 200 -s 0.5
```

target labels are randomly flipped with 5, 10, 15, and 20 bits
```
python -m MultiLabel.JMA_VOC -l 5 -i 200 -s 0.5
python -m MultiLabel.JMA_VOC -l 10 -i 200 -s 0.5
python -m MultiLabel.JMA_VOC -l 15 -i 200 -s 0.5
python -m MultiLabel.JMA_VOC -l 20 -i 200 -s 0.5
```

### Case 3: One-hot
The choosen samples from test dataset is stored in 'Dataset/GTSRB/combined_200_ACCESS_sub_GTSRB_index.npy', while the corresponding
target labels are stored in 'Dataset/GTSRB/combined_200_ACCESS_sub_GTSRB_error_pattern.npy'.

The trained model is stored in 'Model/gtsrb/final_trained_weights.hdf5'.

The demo code could be run as follows:
```
python -m OneHot.JMA_gtsrb_onehot -s 0.5 -i 100
```




