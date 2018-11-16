# Kaggle-Protein-Competition

## Three Notebooks, showing our baseline, Feature Augmentation and U net


### U Net Model


This deep neural network is implemented with Keras functional API, and is based on the python implemenation of the U net paper discussed. 
https://github.com/zhixuhao/unet

Output from the network is a 512*512 which represents mask that should be learned. Sigmoid activation function
makes sure that mask pixels are in \[0, 1\] range.


## Base Line ResNet
The baseline architecture is a ResNet-34 being implemented with fastai. The specific loss function utilized with this model is the focal loss. The focal loss function is designed to be a proper loss function for problems that involves large data imbalances. With 28 label categories and a single label, Nucleoplasm, appearing in over ⅓ of the data entries, it’s helps greatly to have a loss function that is geared towards these kinds of data imbalances. The focal loss essentially does this by focusing on the sparse set of less frequently occurring data and doesn’t allow the well-classified examples to overweight the loss assigned to them.
	
Since a 4-channel input of RGBY is being utilized for a pretrained model accustomed to the normal 3-channel RGB input, the first convolutional layer had to be dropped and replaced with a 7x7 4->64 rather than the previous 7x7 3-> 64. The optimizer for the model used is the Adam Optimizer. The learning rate utilized was 0.02. 
 
 This was found by running training with different learning rates, recording the losses, and finding the learning rate that appears to bring the losses to a minimum. The unique thresholds are utilized for each individual class in order to maximize the accuracy score. 
