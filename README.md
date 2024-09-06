# Handwritten-signature-forgery-detection-using-CNN_in-MATLAB-without-using-Toolbox
The "cnn" folder contains 3 .mat files with the names: startlearning, demoMLCNN, and mlcnn. The startlearning file is used to add the paths of the executed files to the MATLAB direction. Then we save the settings in .config.mat.

Inside the "data" folder, image data is arranged in a pattern. The "visualization" and "utils" folders are for visually displaying the performance of layers and outputs. The main part of the program is located in the "demoMLCNN" and "mlcnn" files. The programming is object-oriented, with our object being "mlcnn." Using the "classdef" command, we create a structure and define parameters and functions necessary for training the network within this structure. Inside this object, all functions such as feedforward, backpropagation, and weight updates are defined.


Inside "mlcnn," we set the initial parameters.

