%%loadDataAlpha
%%Data loader

%%Version 1.0
%%Compatibility: Alpha Frame, Square
%%Author: Thomas DeWitt
%%License: Apache 2.0

function [trainData,testData] = loadDataAlpha()

load vectorizedData train
load vectorizedData test

trainData = prepData(train,true);
testData = prepData(test,false);

end

function newData = prepData(data,vectorize)

dataSize = size(data,1);

newData = cell(dataSize,2); %each row contains data cell and label cell

labels = data(:,1);
images = data(:,2:end);

labels(labels == 0) = 10; %change '0' labels to '10' for indexing
images = images./255; %scale image values for network

for n = 1:dataSize
    newData{n,1} = transpose(reshape(images(n,:),28,28)); %reshape to square for convnet
end

if vectorize == true
    %vectorize label for backprop algorithm
    for n = 1:dataSize
        thisLabel = zeros(10,1);
        thisLabel(labels(n,1)) = 1;
        newData{n,2} = thisLabel;
    end
else
    %keep label
    for n = 1:dataSize
        newData{n,2} = labels(n,1);
    end
end

end