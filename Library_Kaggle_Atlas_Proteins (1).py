
# coding: utf-8

# In[69]:


import numpy as np 
import pandas as pd

from imageio import imread
from skimage.filters import sobel_h,sobel_v
from skimage.measure import block_reduce

import cv2

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from PIL import Image

import tensorflow as tf
from multiprocessing import Pool

import time
import skimage
import os


# In[70]:


label_names = {
    0:  "Nucleoplasm",  
    1:  "Nuclear membrane",   
    2:  "Nucleoli",   
    3:  "Nucleoli fibrillar center",   
    4:  "Nuclear speckles",
    5:  "Nuclear bodies",   
    6:  "Endoplasmic reticulum",   
    7:  "Golgi apparatus",   
    8:  "Peroxisomes",   
    9:  "Endosomes",   
    10:  "Lysosomes",   
    11:  "Intermediate filaments",   
    12:  "Actin filaments",   
    13:  "Focal adhesion sites",   
    14:  "Microtubules",   
    15:  "Microtubule ends",   
    16:  "Cytokinetic bridge",   
    17:  "Mitotic spindle",   
    18:  "Microtubule organizing center",   
    19:  "Centrosome", 
    20:  "Lipid droplets",   
    21:  "Plasma membrane",   
    22:  "Cell junctions",   
    23:  "Mitochondria",   
    24:  "Aggresome",   
    25:  "Cytosol",   
    26:  "Cytoplasmic bodies",   
    27:  "Rods & rings"
}


# In[71]:


reverse_train_labels = dict((v,k) for k,v in label_names.items())


# In[67]:


def get_label(file_ids, train_labels):
    l=train_labels[train_labels.Id.isin(file_ids)]["Target"].values
    l= [[int(i) for i in s] for s in l]
    labels = []
    for i in l:
        L = [0]*28
        for j in i:
            L[j]=1
        labels.append(L)
    return np.array(labels)


# In[4]:


def fill_targets(row):
    row.Target = np.array(row.Target.split(" ")).astype(np.int)
    for num in row.Target:
        name = label_names[int(num)]
        row.loc[name] = 1
    return row


# In[5]:


def grad(image):
    grad_x0=sobel_h(image[:,:,0])
    grad_y0=sobel_v(image[:,:,0])
    grad0=np.sqrt(grad_x0*grad_x0+grad_y0*grad_y0).T
    
    grad_x1=sobel_h(image[:,:,1])
    grad_y1=sobel_v(image[:,:,1])
    grad1=np.sqrt(grad_x1*grad_x1+grad_y1*grad_y1).T
    
    grad_x2=sobel_h(image[:,:,2])
    grad_y2=sobel_v(image[:,:,2])
    grad2=np.sqrt(grad_x2*grad_x2+grad_y2*grad_y2).T
    
    grad_x3=sobel_h(image[:,:,3])
    grad_y3=sobel_v(image[:,:,3])
    grad3=np.sqrt(grad_x3*grad_x3+grad_y3*grad_y3).T
    
    return np.array([grad0,grad1,grad2, grad3]).T

def grad_threshold(image, eps):
    
    return (grad(image) > eps)*255


# In[6]:


def dirrectional_grad(image,theta):
    
    grad_x0 = np.cos(theta)*sobel_h(image[:,:,0]) + np.sin(theta)*sobel_v(image[:,:,0])
    grad_x1 = np.cos(theta)*sobel_h(image[:,:,1]) + np.sin(theta)*sobel_v(image[:,:,1])
    grad_x2 = np.cos(theta)*sobel_h(image[:,:,2]) + np.sin(theta)*sobel_v(image[:,:,2])
    grad_x3 = np.cos(theta)*sobel_h(image[:,:,3]) + np.sin(theta)*sobel_v(image[:,:,3])
    
    grad0= np.maximum(grad_x0,0).T
    grad1= np.maximum(grad_x1,0).T
    grad2= np.maximum(grad_x2,0).T
    grad3= np.maximum(grad_x3,0).T
    image= np.array([grad0,grad1,grad2, grad3]).T
    return image



# In[7]:


def load_image(basepath, image_id):

    images = np.zeros(shape=(512,512,4))
    images[:,:,0] = imread(basepath + "/" + image_id + "_green" + ".png")
    images[:,:,1] = imread(basepath + "/" + image_id + "_red" + ".png")
    images[:,:,2] = imread(basepath + "/" + image_id + "_blue" + ".png")
    images[:,:,3] = imread(basepath + "/" + image_id + "_yellow" + ".png")

    return images

    
def make_title(file_id, train_labels):
    file_targets = train_labels.loc[train_labels.Id==file_id, "Target"].values[0]
    title = " - "
    for n in file_targets:
        title += label_names[n] + " - "
    return title


# In[9]:


def return_name(img):
    name=[]
    l=list(label_names.values())
    j=0
    for i in labels[img]:
        if i==1:
            name.append(l[j])
        j+=1
    return name


# In[10]:


def make_image_row(image, subax, title, i ):
    subax[0].imshow(image[:,:,4*i], cmap="Greens")
    subax[1].imshow(image[:,:,4*i+1], cmap="Reds")
    subax[1].set_title("stained microtubules")
    subax[2].imshow(image[:,:,4*i+2], cmap="Blues")
    subax[2].set_title("stained nucleus")
    subax[3].imshow(image[:,:,4*i+3], cmap="Oranges")
    subax[3].set_title("stained endoplasmatic reticulum")
    subax[0].set_title(title)
    return subax


# In[3]:


class TargetGroupIterator:
    
    def __init__(self, target_names, batch_size, labels_path, basepath, ws, grad, dir_grad, grad_threshold, nb_threshold, nb_rot, reduce, block_size):
        
        self.target_names = target_names
        self.target_list = [reverse_train_labels[key] for key in self.target_names]
        self.batch_shape = (batch_size, 512, 512, 4)
        self.basepath = basepath
        self.train_labels = pd.read_csv(labels_path)
        self.train_labels = self.train_labels.apply(fill_targets, axis=1)
        self.ws = ws
        self.grad = grad
        self.dir_grad = dir_grad
        self.grad_threshold = grad_threshold
        self.nb_threshold = nb_threshold 
        self.nb_rot = nb_rot
        self.reduce = reduce
        self.block_size = block_size
    
    def reduce(self, images):
        block=(1,block_size,block_size,1)
        return block_reduce(images,block,np.mean)
        
    def features_aumentation(self, image):    

        grad_=self.grad
        dir_grad_=self.dir_grad
        grad_threshold_=self.grad_threshold

        a=image

        if grad_:
            a=np.append(a,grad(image), axis=2)

        if dir_grad_:
            rot=np.arange(0, 360, 360//self.nb_rot)
            for i in rot:
                a=np.append(a,dirrectional_grad(image,i), axis=2)

        if grad_threshold_:
            eps=np.arange(0,128,128//self.nb_threshold)
            for e in eps:      
                a=np.append(a, grad_threshold(image,e), axis=2)

        return a
    
    def find_matching_data_entries(self):

        self.train_labels["check_col"] = self.train_labels.Target.apply(
            lambda l: self.check_subset(l)
        )
        self.images_identifier = self.train_labels[self.train_labels.check_col==1].Id.values
        self.train_labels.drop("check_col", axis=1, inplace=True)
    
    def check_subset(self, targets):
        return np.where(set(self.target_list).issuperset(set(targets)), 1, 0)
    
    def get_loader(self):

        filenames = []
        idx = 0
        images = np.zeros(self.batch_shape)
        p = Pool(processes=self.ws)
        for image_id in self.images_identifier:
            images[idx,:,:,:] = load_image(self.basepath, image_id)
            filenames.append(image_id)
            idx += 1
            if idx == self.batch_shape[0]:
                images = np.array(p.map(self.features_aumentation, images))
                if self.reduce:
                    yield filenames, get_label(filenames,self.train_labels), reduce(images)
                else:
                    yield filenames, get_label(filenames,self.train_labels), images
                
                filenames = []
                images = np.zeros(self.batch_shape)
                idx = 0
        
        if idx > 0:

            images = np.array(p.map(self.features_aumentation, images))
            if self.reduce:
                yield filenames, get_label(filenames,self.train_labels), reduce(images)
            else:
                yield filenames, get_label(filenames,self.train_labels), images
        p.close()

