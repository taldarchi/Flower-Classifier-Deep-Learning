# Flower-Classifier-Deep-Learning
TRANSFER LEARNING: A deep network realization project in Linux

# Introduction
In this project i implemented a neural network to classify colored images of flowers from seventeen different types using the TRANSFER LEARNING technique:
* Uploaded from a network file designed to classify 1.2 million images from 1000 types (ILSVRC competition)
created by Google in the GoogLeNet_v2_nn.t7 file. This file contains the network structure and all the parameters
Trained in a separate process.
* A new network has been created that includes the first 10 layers of the GoogLeNet_v2_nn.t7 network
also added are a number of layers to the new grid for the classification of flower pictures.
* Train the new network while freezing the parameters of the first 10 layers. That is, only the parameters of the new layers
will be trained, using the image training set provided in the flowers.t7 file.
* Examination of the network performance we have learned about the flower image test set.


# Network:
The attached files - transfer_learning_project.lua file loads all images and creates a 17-group TENSOR
Flowers, and in each group 80 color images in size 128 X 128 pixels. In addition, a training set (85% of the total) is created
(Including 15% of all images), each with 17 groups.

The attached transfer_learning_project file loads GoogLeNet_v2_nn.t7, and creates a new neural network that includes
The first 10 layers of the GoogLeNet network. In addition, the code implements the freezing of the parameters of these layers.

The following components were added to the new neural network:
1. Convolution layer with input dimension 320, output dimension 16, filter size 3X3, Stride = 1, without PADDING.
2. RELU
3. MAXPOOLING on 4x4 windows with Stride = 4 in each dimension.
4. Convert the FEATURE MAPS to the column vector (using nn.View)
5. DROPOUT with 50% probability (using nn.Dropout.)
6. FULLY CONNECTED (using nn.Linear, the number of ports is NUM_CLASSES)
7. LOGSOFTMAX

Added a view of the chance of error (training and testing) using LOGGER and CONFUSION MATRIX.
MiniBatch is set to 32, training algorithm is ADAM, learning rate = 0.1.
maximum number of training steps is set to 200, Early Stopping is applied (less than 10%).


# How To:
Download flowers.t7 (https://drive.google.com/open?id=0B08krT1KTfJMc2wtcDJWcElqR1E) and GoogLeNet_v2_nn (https://drive.google.com/open?id=0B08krT1KTfJMYjRhSW1pUkV1ZmM) and put it in the same folder as the file transfer_learning_project.lua and run it.
Set NUM_CLASSES = [Number of flower types]
