load 'signature.mat';
load 'signaturetest.mat';
trainData = reshape(pattern,200,200,1,400);
trainLabels = double(target');
testData = reshape(patterntest,200,200,1,100);
testLabels = double(targettest');

dataSize = [200,200,1];

arch = {struct('type','input','dataSize',dataSize), ...
        struct('type','conv','filterSize',[5 5], 'nFM', 6), ...
        struct('type','subsample','stride',[2 2]), ...
        struct('type','conv','lRate',0.01,'filterSize',[4 5], 'nFM',12,'actFun','tanh'), ...
        struct('type','subsample','stride',2), ...
        struct('type','hidden','lRate',0.000001,'nFM',50,'actFun','tanh'),...
        struct('type','output', 'nOut', 2)};

n = mlcnn(arch);
n.batchSize = 100;
n.costFun = 'xent';
n.nEpoch = 2;

n = n.train(trainData,trainLabels);
clear trainData trainLabels;

classErr = n.test(testData,testLabels,'classerr');
close all

figure('Name','Learned Layer Filters');
fprintf('\nDisplaying Layer Filters...\n');
visMLCNNFilters(n,[-.4 .4]) 

figure('Name','Layer Feature Maps');
fprintf('\nDisplaying Feature Maps...\n');
visMLCNNFeatureMaps(n); colormap gray



