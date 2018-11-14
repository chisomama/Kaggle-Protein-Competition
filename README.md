# Kaggle-Protein-Competition

## Three Notebooks, showing our baseline, Feature Augmentation and U net


### U Net Model


This deep neural network is implemented with Keras functional API, and is based on the python implemenation of the U net paper discussed. 
https://github.com/zhixuhao/unet

Output from the network is a 512*512 which represents mask that should be learned. Sigmoid activation function
makes sure that mask pixels are in \[0, 1\] range.
