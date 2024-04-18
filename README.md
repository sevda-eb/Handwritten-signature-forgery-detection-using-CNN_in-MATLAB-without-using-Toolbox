# Handwritten-signature-forgery-detection-using-CNN_in-MATLAB-without-using-Toolbox
The "cnn" folder contains 3 .mat files with the names: startlearning, demoMLCNN, and mlcnn. The startlearning file is used to add the paths of the executed files to the MATLAB direction. Then we save the settings in .config.mat.

Inside the "data" folder, image data is arranged in a pattern. The "visualization" and "utils" folders are for visually displaying the performance of layers and outputs. The main part of the program is located in the "demoMLCNN" and "mlcnn" files. The programming is object-oriented, with our object being "mlcnn." Using the "classdef" command, we create a structure and define parameters and functions necessary for training the network within this structure. Inside this object, all functions such as feedforward, backpropagation, and weight updates are defined.


Inside "mlcnn," we set the initial parameters.

First, we use the "load" command to load the data. Our pattern will be a matrix of size 400*40000, where 400 is the number of samples we have, and each sample has a size of 40000. Since CNN inputs must be in a two-dimensional format like an image, we define the 40000-dimensional pattern as a 2D array of size 200*200.

Because our image is black and white, the depth of our data is 1. However, if it were, for example, a color image, our data depth would be applied to the input neurons like a cube.

Next, we enter the "demoMLCNN" section. This is where the main structure and architecture of the convolutional network are determined. The network consists of an input layer, two convolutional layers, and two pooling layers, arranged such that after each convolutional layer, there is a pooling layer.

After the last pooling layer, we have a hidden layer, followed by the output layer. The structure and number of neurons in each layer are specified in the following code snippets:

% Input layer
net.layers{1} = struct('type', 'input', 'mapsize', [44 44], 'depth', 1);

% First convolutional layer
net.layers{2} = struct('type', 'conv', 'filtersize', [5 5], 'numfilters', 6, 'stride', 1, 'pad', 0);

% First pooling layer
net.layers{3} = struct('type', 'pool', 'poolsize', [2 2], 'stride', 2);

% Second convolutional layer
net.layers{4} = struct('type', 'conv', 'filtersize', [5 5], 'numfilters', 12, 'stride', 1, 'pad', 0);

% Second pooling layer
net.layers{5} = struct('type', 'pool', 'poolsize', [2 2], 'stride', 2);

% Hidden layer
net.layers{6} = struct('type', 'fc', 'numneurons', 100);

% Output layer
net.layers{7} = struct('type', 'output', 'numclasses', 10);

In this code snippet, "mapsize" refers to the size of the feature map, "filtersize" is the size of the filters, "numfilters" is the number of filters, "poolsize" is the size of the pooling region, "numneurons" is the number of neurons in the hidden layer, and "numclasses" is the number of classes in the output layer.

