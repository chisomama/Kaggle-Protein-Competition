# Kaggle-Protein-Competition

## Three Notebooks, showing our baseline, Feature Augmentation and U net


### U Net Model


This deep neural network is implemented with Keras functional API, and is based on the python implemenation of the U net paper discussed. 
https://github.com/zhixuhao/unet

Output from the network is a 512*512 which represents mask that should be learned. Sigmoid activation function
makes sure that mask pixels are in \[0, 1\] range.


# Data Processing Folder

#### .      Feature_Augmentation.ipynb
            -This notebook is our explroation of our feature augmentation exploration and the eventual specific feature augmentations that were implemented. The eventual feature augmentations that were implemented include normaed gradient using Sobel kernel, directional gradient, normed gradient using Sobel kernel and thresholds, and feature engineering with blocks. 

#### .      Data_Augmentation.ipynb
            -This notebook covers the data augmentation that was used during the project. With the large data skew provided in the training data our team attempted to augment the underpresented data. This was performed by subjecting all targets with less than 100 occurences to 11 data augmentations resulting in the generation of over 250,000 unique additional training images. The augmentations ranged from combinations of flips and rotations.


####        EplorationAndBaseline.ipynb
		-A notebook provided by an outside user Allunia onto the Kaggle website. The notebook was found to be incredibly useful in order to get a firm understanding of the data provided for the Protein Atlas Kaggle competition and helped influence our team in furthering our research.

## Base Line ResNet
The baseline architecture is a ResNet-34 being implemented with fastai. The specific loss function utilized with this model is the focal loss. The focal loss function is designed to be a proper loss function for problems that involves large data imbalances. With 28 label categories and a single label, Nucleoplasm, appearing in over ⅓ of the data entries, it’s helps greatly to have a loss function that is geared towards these kinds of data imbalances. The focal loss essentially does this by focusing on the sparse set of less frequently occurring data and doesn’t allow the well-classified examples to overweight the loss assigned to them.
	
Since a 4-channel input of RGBY is being utilized for a pretrained model accustomed to the normal 3-channel RGB input, the first convolutional layer had to be dropped and replaced with a 7x7 4->64 rather than the previous 7x7 3-> 64. The optimizer for the model used is the Adam Optimizer. The learning rate utilized was 0.02. 
 
 This was found by running training with different learning rates, recording the losses, and finding the learning rate that appears to bring the losses to a minimum. The unique thresholds are utilized for each individual class in order to maximize the accuracy score. 
